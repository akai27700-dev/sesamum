import os
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional

class ONNXInference:
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.session = None
        self.input_name = None
        self.output_names = None
        self.active_provider = "CPUExecutionProvider"
        self._dll_dirs = []
        self._load_model()

    def _prepare_windows_cuda_dlls(self):
        if os.name != "nt":
            return
        candidate_dirs = []
        venv_site = os.path.abspath(os.path.join(os.path.dirname(self.model_path), "..", ".venv", "Lib", "site-packages"))
        candidate_dirs.extend([
            os.path.join(venv_site, "nvidia", "cuda_runtime", "bin"),
            os.path.join(venv_site, "nvidia", "cublas", "bin"),
            os.path.join(venv_site, "nvidia", "cuda_nvrtc", "bin"),
            os.path.join(venv_site, "torch", "lib"),
        ])
        for path in candidate_dirs:
            if os.path.isdir(path):
                try:
                    self._dll_dirs.append(os.add_dll_directory(path))
                except (FileNotFoundError, OSError):
                    pass
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
    
    def _load_model(self):
        """ONNXモデルをロード"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self._prepare_windows_cuda_dlls()
        
        # 実行環境を設定
        available_providers = set(ort.get_available_providers())
        if self.use_gpu and 'CUDAExecutionProvider' in available_providers:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']

        # セッションを作成
        session_options = ort.SessionOptions()
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers,
            sess_options=session_options
        )
        
        # 入出力名を取得
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.active_provider = self.session.get_providers()[0] if self.session.get_providers() else "CPUExecutionProvider"
        
        print(f"ONNX model loaded: {self.model_path}")
        print(f"Providers: {self.session.get_providers()}")

    def get_runtime_device(self) -> str:
        provider = str(self.active_provider)
        return "cuda" if "CUDAExecutionProvider" in provider else "cpu"
    
    def infer_batch(self, input_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # 入力をfloat32に変換
        if input_batch.dtype != np.float32:
            input_batch = input_batch.astype(np.float32)
        
        # 推論実行
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_batch}
        )
        
        policy, value = outputs
        
        # ONNXモデルがsoftmaxしていない場合は方策確率をsoftmaxで正規化
        if policy.max() > 1.0 or policy.min() < 0.0:
            # softmax変換が必要
            policy_exp = np.exp(policy - np.max(policy, axis=1, keepdims=True))
            policy = policy_exp / np.sum(policy_exp, axis=1, keepdims=True)
        
        return policy, value
    
    def infer_single(self, p_board: int, o_board: int, turn: int) -> Tuple[np.ndarray, float]:

        from .othello_core import make_input_tensor
        
        # 入力テンソルを作成
        tensor_batch = make_input_tensor([(p_board, o_board, turn)])
        
        # 推論実行
        policy_batch, value_batch = self.infer_batch(tensor_batch)
        
        return policy_batch[0], float(value_batch[0, 0])
    
    def get_model_info(self) -> dict:
        """モデル情報を取得"""
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()
        
        return {
            'input_shape': input_info.shape,
            'input_type': input_info.type,
            'output_shapes': [output.shape for output in output_info],
            'output_types': [output.type for output in output_info],
            'providers': self.session.get_providers()
        }

def convert_pytorch_to_onnx(pytorch_model, output_path: str, input_shape: Tuple[int, ...] = (1, 3, 8, 8)):

    import torch
    
    model = pytorch_model.eval()
    
    # ダミー入力を作成
    dummy_input = torch.randn(input_shape)
    
    # ONNXにエクスポート（opset_versionを18に変更）
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        dynamo=False,
        opset_version=18,  # opset_versionを18に変更
        do_constant_folding=True,
        input_names=['input'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    
    print(f"PyTorch model converted to ONNX: {output_path}")

def create_onnx_model_from_pytorch(pytorch_model_path: str, onnx_output_path: str):

    import torch
    from .othello_core import OthelloNet
    
    # PyTorchモデルをロード
    model = OthelloNet()
    if os.path.exists(pytorch_model_path):
        model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu', weights_only=True))
        print(f"PyTorch model loaded: {pytorch_model_path}")
    else:
        print(f"PyTorch model not found: {pytorch_model_path}, using random weights")
    
    # ONNXに変換
    convert_pytorch_to_onnx(model, onnx_output_path)
    
    return onnx_output_path
