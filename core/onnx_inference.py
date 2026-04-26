import os
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional

class ONNXInference:
    """ONNXモデルを使用した高速推論エンジン"""
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.session = None
        self.input_name = None
        self.output_names = None
        self._load_model()
    
    def _load_model(self):
        """ONNXモデルをロード"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        
        # 動的なスレッド数設定
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        # SessionOptions で詳細設定
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # CPUコア数に応じてスレッド数を動的に設定（高性能CPU向けに最適化）
        if cpu_count >= 32:
            intra_threads = min(cpu_count, 32)  # 超ハイパフォーマンスCPU：32スレッドまで
        elif cpu_count >= 24:
            intra_threads = min(cpu_count, 24)  # ハイパフォーマンスCPU：24スレッドまで
        elif cpu_count >= 16:
            intra_threads = min(cpu_count, 20)  # ミドルハイCPU：20スレッドまで
        elif cpu_count >= 12:
            intra_threads = min(cpu_count, 16)  # ミドルレンジCPU：16スレッドまで
        elif cpu_count >= 8:
            intra_threads = cpu_count           # 8-11コア：全コア使用
        elif cpu_count >= 4:
            intra_threads = cpu_count           # 4-7コア：全コア使用
        else:
            intra_threads = max(1, cpu_count)   # 1-3コア：1スレッドでも安定
            
        sess_opts.intra_op_num_threads = intra_threads
        sess_opts.inter_op_num_threads = max(1, intra_threads // 2)  # 演算間の並列化も有効化
        
        # 実行環境を設定
        if self.use_gpu:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB に増加
                    'do_copy_in_default_stream': True,
                    'tunable_op_enable': False,
                }),
                'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']
        
        # セッションを作成
        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers,
            sess_options=sess_opts
        )
        
        # 入出力名を取得
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"ONNX model loaded: {self.model_path}")
        print(f"Providers: {self.session.get_providers()}")
    
    def infer_batch(self, input_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        バッチ推論を実行
        
        Args:
            input_batch: (batch_size, 3, 8, 8) の入力テンソル
            
        Returns:
            (policy, value) のタプル
            policy: (batch_size, 64) の方策確率
            value: (batch_size, 1) の価値
        """
        # 入力をfloat32に変換
        if input_batch.dtype != np.float32:
            input_batch = input_batch.astype(np.float32)
        
        # CPUコア数に応じて動的にバッチサイズを調整（高性能CPU向けに最適化）
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        if cpu_count >= 32:
            max_batch_size = 4096  # 超ハイパフォーマンスCPU
        elif cpu_count >= 24:
            max_batch_size = 3072  # ハイパフォーマンスCPU
        elif cpu_count >= 16:
            max_batch_size = 2048  # ミドルハイCPU
        elif cpu_count >= 12:
            max_batch_size = 1536  # ミドルレンジCPU
        elif cpu_count >= 8:
            max_batch_size = 1024  # 一般的なCPU
        elif cpu_count >= 4:
            max_batch_size = 512   # 4-7コア：中程度のバッチ
        else:
            max_batch_size = 256   # 1-3コア：小さなバッチで安定
            
        # 大きなバッチの場合は分割処理
        if len(input_batch) > max_batch_size:
            policies = []
            values = []
            for i in range(0, len(input_batch), max_batch_size):
                batch_slice = input_batch[i:i+max_batch_size]
                policy, value = self.infer_batch(batch_slice)
                policies.append(policy)
                values.append(value)
            return np.vstack(policies), np.vstack(values)
        
        # 推論実行
        try:
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_batch}
            )
        except Exception as e:
            # ONNX メモリ不足時のエラーハンドリング
            print(f"ONNX inference error: {e}")
            raise
        
        policy, value = outputs
        
        # 方策確率をsoftmaxで正規化（ONNXモデルがsoftmaxしていない場合）
        if policy.max() > 1.0 or policy.min() < 0.0:
            # softmax変換が必要
            policy_exp = np.exp(policy - np.max(policy, axis=1, keepdims=True))
            policy = policy_exp / np.sum(policy_exp, axis=1, keepdims=True)
        
        return policy, value
    
    def infer_single(self, p_board: int, o_board: int, turn: int) -> Tuple[np.ndarray, float]:
        """
        単一局面の推論
        
        Args:
            p_board: プレイヤーのビットボード
            o_board: 相手のビットボード  
            turn: 手番 (1 or -1)
            
        Returns:
            (policy, value) のタプル
            policy: (64,) の方策確率
            value: 価値
        """
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
    """
    PyTorchモデルをONNXに変換
    
    Args:
        pytorch_model: PyTorchモデル
        output_path: 出力先ONNXファイルパス
        input_shape: 入力テンソルの形状
    """
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
    """
    PyTorchモデルファイルからONNXモデルを作成
    
    Args:
        pytorch_model_path: PyTorchモデルファイルパス
        onnx_output_path: 出力先ONNXファイルパス
    """
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
