import tkinter as tk

class StartupSettingsDialog:

    def __init__(self, parent, cpp_available, nn_available=True, nn_reason=''):
        self.result = None
        self.top = tk.Toplevel(parent)
        self.top.title('探索設定')
        self.top.transient(parent)
        self.top.grab_set()
        self.top.resizable(False, False)
        self.cpp_available = cpp_available
        self.nn_available = nn_available
        self.nn_reason = nn_reason
        self.use_cpp = tk.BooleanVar(value=True)
        self.use_nn = tk.BooleanVar(value=True)
        self.use_mcts = tk.BooleanVar(value=True)
        self.use_tt = tk.BooleanVar(value=True)
        self.search_mode = tk.StringVar(value='hybrid')
        self.book_source = tk.StringVar(value='egaroucid')
        self.book_usage = tk.IntVar(value=85)
        self.time_limit = tk.StringVar(value='5')
        self.player_color = tk.StringVar(value='black')
        self.use_pondering = tk.BooleanVar(value=True)
        
        # 枝刈り設定
        self.pruning_enabled = tk.BooleanVar(value=True)
        self.mcts_pruning_enabled = tk.BooleanVar(value=False)
        self.mcts_pruning_time = tk.DoubleVar(value=2.0)
        self.mcts_pruning_branches = tk.IntVar(value=3)
        self.ab_pruning_time = tk.DoubleVar(value=3.0)
        self.traditional_pruning_enabled = tk.BooleanVar(value=True)
        self.multi_cut_enabled = tk.BooleanVar(value=True)
        self.multi_cut_threshold = tk.IntVar(value=3)
        self.multi_cut_depth = tk.IntVar(value=8)
        frame = tk.Frame(self.top, padx=16, pady=14)
        frame.pack(fill='both', expand=True)
        
        # 左列
        left_frame = tk.Frame(frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 4))
        
        # 右列
        right_frame = tk.Frame(frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=(4, 0))
        
        # 左列の設定項目
        self.cpp_checkbox = self._add_checkbox(left_frame, 'C++探索', self.use_cpp, state='normal' if cpp_available else 'disabled')
        self.nn_checkbox = self._add_checkbox(left_frame, 'NN', self.use_nn, state='normal' if nn_available else 'disabled')
        self.mcts_influence_scale = self._add_radio_group(left_frame, '探索モード', self.search_mode, [('αβ+MCTS', 'hybrid'), ('αβのみ', 'ab_only'), ('MCTSのみ', 'mcts_only')])
        self.tt_checkbox = self._add_checkbox(left_frame, 'TT再利用', self.use_tt)
        self.pondering_checkbox = self._add_checkbox(left_frame, '常に思考', self.use_pondering)
        self._add_radio_group(left_frame, '定石ソース', self.book_source, [('JSON', 'json'), ('Egaroucid', 'egaroucid'), ('使用しない', 'none')])
        self.book_usage_scale = self._add_scale_group(left_frame, '定石使用確率(JSONのみ)', self.book_usage, 0, 100)
        self._add_radio_group(left_frame, '最大思考時間', self.time_limit, [('1秒', '0.5'), ('5秒', '5'), ('10秒', '10'), ('30秒', '30')])
        self._add_radio_group(left_frame, 'プレイヤー', self.player_color, [('黒', 'black'), ('白', 'white')])
        
        # 右列の枝刈り設定
        self.pruning_checkbox = self._add_checkbox(right_frame, '枝刈り有効', self.pruning_enabled)
        self.traditional_pruning_checkbox = self._add_checkbox(right_frame, '従来枝刈り', self.traditional_pruning_enabled)
        self.mcts_pruning_checkbox = self._add_checkbox(right_frame, 'MCTS枝刈り', self.mcts_pruning_enabled)
        self.mcts_time_scale = self._add_scale_group(right_frame, 'MCTS事前探索時間(秒)', self.mcts_pruning_time, 0.5, 10.0, 0.5)
        self.mcts_branches_scale = self._add_scale_group(right_frame, 'MCTS選択枝数', self.mcts_pruning_branches, 1, 8, 1)
        self.ab_time_scale = self._add_scale_group(right_frame, 'AB探索時間(秒)', self.ab_pruning_time, 0.5, 15.0, 0.5)
        self.multi_cut_checkbox = self._add_checkbox(right_frame, 'マルチカット', self.multi_cut_enabled)
        self.multi_cut_threshold_scale = self._add_scale_group(right_frame, 'マルチカット閾値', self.multi_cut_threshold, 2, 6)
        self.multi_cut_depth_scale = self._add_scale_group(right_frame, 'マルチカット深さ', self.multi_cut_depth, 4, 12)
        
        # startボタンを右列に配置
        start_btn = tk.Button(right_frame, text='start', command=self.on_ok, width=10)
        start_btn.pack(side='right', pady=(8, 0))
        self.use_nn.trace_add('write', self._toggle_nn_options)
        self.search_mode.trace_add('write', self._toggle_search_mode_options)
        self.book_source.trace_add('write', self._toggle_book_options)
        self._toggle_nn_options()
        self._toggle_book_options()
        self._toggle_mcts_pruning_options()
        if not cpp_available:
            tk.Label(left_frame, text='C++エンジンが未ロードのため Python 実装で動作します。', anchor='w', fg='#b71c1c').pack(fill='x', pady=(4, 8))
        if not nn_available:
            msg = 'NN機能は利用できないため αβ のみで動作します。'
            if nn_reason:
                msg = f'{msg} {nn_reason}'
            tk.Label(left_frame, text=msg, anchor='w', fg='#b71c1c', wraplength=200, justify='left').pack(fill='x', pady=(0, 8))
        
        self.top.protocol('WM_DELETE_WINDOW', self.on_ok)
        self.top.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.top.winfo_width()) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.top.winfo_height()) // 2)
        self.top.geometry(f'+{x}+{y}')
        self.top.wait_window()

    def _add_radio_group(self, parent, label, variable, options, state='normal'):
        box = tk.LabelFrame(parent, text=label, padx=8, pady=6)
        box.pack(fill='x', pady=(0, 8))
        for text, value in options:
            tk.Radiobutton(box, text=text, value=value, variable=variable, state=state).pack(side='left', padx=(0, 10))
        return box

    def _add_scale_group(self, parent, label, variable, min_value, max_value, resolution=1):
        box = tk.LabelFrame(parent, text=label, padx=8, pady=6)
        box.pack(fill='x', pady=(0, 8))
        scale = tk.Scale(box, from_=min_value, to=max_value, orient='horizontal', variable=variable, resolution=resolution)
        scale.pack(fill='x')
        return scale

    def _add_checkbox(self, parent, label, variable, state='normal'):
        box = tk.LabelFrame(parent, text=label, padx=8, pady=6)
        box.pack(fill='x', pady=(0, 8))
        checkbox = tk.Checkbutton(box, text='ON', variable=variable, state=state)
        checkbox.pack(side='left')
        return checkbox

    def _toggle_nn_options(self, *_):
        nn_on = self.use_nn.get()
        mode = self.search_mode.get()
        if not nn_on and mode == 'mcts_only':
            self.search_mode.set('ab_only')
        for child in self.mcts_influence_scale.winfo_children():
            if isinstance(child, tk.Radiobutton):
                child.config(state='normal' if nn_on else 'disabled')
        # MCTS枝刈りオプションも更新
        self._toggle_mcts_pruning_options()

    def _toggle_search_mode_options(self, *_):
        """探索モード変更時にMCTS枝刈りオプションを更新"""
        # NNオプションと同様の処理を行う
        self._toggle_nn_options()

    def _toggle_book_options(self, *_):
        book_enabled = self.book_source.get() != 'none'
        book_json = self.book_source.get() == 'json'
        self.book_usage_scale.config(state='normal' if book_enabled and book_json else 'disabled')

    def _toggle_mcts_pruning_options(self, *_):
        """MCTSが無効なときにMCTS関連の枝刈り設定を無効化"""
        nn_on = self.use_nn.get()
        mode = self.search_mode.get()
        mcts_enabled = nn_on and mode != 'ab_only'
        
        # MCTS枝刈り関連の設定を制御
        mcts_pruning_state = 'normal' if mcts_enabled else 'disabled'
        
        # MCTS枝刈りチェックボックス
        if hasattr(self, 'mcts_pruning_checkbox'):
            self.mcts_pruning_checkbox.config(state=mcts_pruning_state)
        
        # MCTS関連のスケール
        if hasattr(self, 'mcts_time_scale'):
            self.mcts_time_scale.config(state=mcts_pruning_state)
        if hasattr(self, 'mcts_branches_scale'):
            self.mcts_branches_scale.config(state=mcts_pruning_state)
        
        # MCTSが無効な場合はMCTS枝刈りをOFFに
        if not mcts_enabled:
            self.mcts_pruning_enabled.set(False)

    def on_ok(self):
        nn_enabled = self.use_nn.get()
        mode = self.search_mode.get()
        book_usage_pct = self.book_usage.get()
        time_val = self.time_limit.get()
        time_limit = float(time_val)
        use_mcts = nn_enabled and mode != 'ab_only'
        use_mcts_only = nn_enabled and mode == 'mcts_only'
        mcts_influence = 0 if mode == 'ab_only' else 100 if mode == 'mcts_only' else 50
        self.result = {
            'use_cpp': self.use_cpp.get() and self.cpp_available, 
            'use_nn': nn_enabled, 
            'use_mcts': use_mcts, 
            'search_mode': mode, 
            'use_mcts_only': use_mcts_only, 
            'mcts_influence': mcts_influence, 
            'book_source': self.book_source.get(), 
            'book_usage': book_usage_pct if self.book_source.get() == 'json' else 0, 
            'use_book': self.book_source.get() != 'none', 
            'use_tt': self.use_tt.get(), 
            'time_limit': time_limit, 
            'player_color': self.player_color.get(), 
            'auto_time': True, 
            'auto_mode_type': 'normal', 
            'use_pondering': self.use_pondering.get(),
            # 枝刈り設定
            'pruning_enabled': self.pruning_enabled.get(),
            'mcts_pruning_enabled': self.mcts_pruning_enabled.get(),
            'mcts_pruning_time': self.mcts_pruning_time.get(),
            'mcts_pruning_branches': self.mcts_pruning_branches.get(),
            'ab_pruning_time': self.ab_pruning_time.get(),
            'traditional_pruning_enabled': self.traditional_pruning_enabled.get(),
            'multi_cut_enabled': self.multi_cut_enabled.get(),
            'multi_cut_threshold': self.multi_cut_threshold.get(),
            'multi_cut_depth': self.multi_cut_depth.get()
        }
        self.top.destroy()
