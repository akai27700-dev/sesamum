import tkinter as tk

class StartupSettingsDialog:

    def __init__(self, parent, cpp_available, nn_available=True, nn_reason=''):
        self.result = None
        self.top = tk.Toplevel(parent)
        self.top.title('Sesamum')
        self.top.transient(parent)
        self.top.grab_set()
        self.top.resizable(False, False)
        self.cpp_available = cpp_available
        self.nn_available = nn_available
        self.nn_reason = nn_reason
        
        # 変数の初期化
        self.use_cpp = tk.BooleanVar(value=True)
        self.use_nn = tk.BooleanVar(value=True)
        self.use_mcts = tk.BooleanVar(value=True)
        self.use_tt = tk.BooleanVar(value=True)
        self.search_mode = tk.StringVar(value='hybrid')
        self.book_source = tk.StringVar(value='egaroucid')
        self.book_usage = tk.IntVar(value=85)
        self.time_limit = tk.StringVar(value='5.0') # 秒
        self.player_color = tk.StringVar(value='black')
        self.use_pondering = tk.BooleanVar(value=True)
        self.exact_threshold = tk.IntVar(value=24) # 読み切り開始しきい値
        
        # 枝刈り・最適化設定
        self.pruning_level = tk.StringVar(value='aggressive') # extreme, aggressive, standard, none
        self.mcts_pruning_time = tk.DoubleVar(value=2.0)
        self.mcts_pruning_branches = tk.IntVar(value=3)
        self.ab_pruning_time = tk.DoubleVar(value=3.0)
        self.multi_cut_threshold = tk.IntVar(value=3)
        self.multi_cut_depth = tk.IntVar(value=8)

        # メインフレーム
        main_frame = tk.Frame(self.top, padx=20, pady=20, bg='#f8fafc')
        main_frame.pack(fill='both', expand=True)
        
        # ヘッダー
        header_label = tk.Label(main_frame, text='動作設定', font=('Yu Gothic UI', 14, 'bold'), bg='#f8fafc', fg='#1e293b')
        header_label.pack(anchor='w', pady=(0, 15))

        content_frame = tk.Frame(main_frame, bg='#f8fafc')
        content_frame.pack(fill='both', expand=True)
        
        # 左列: 基本設定
        left_col = tk.Frame(content_frame, bg='#f8fafc')
        left_col.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # 右列: 詳細設定
        right_col = tk.Frame(content_frame, bg='#f8fafc')
        right_col.pack(side='left', fill='both', expand=True, padx=(10, 0))

        # --- 左列項目 ---
        # エンジン実装
        self._add_radio_pair(left_col, 'エンジン実装', self.use_cpp, [('C++', True), ('Python', False)], 
                            state='normal' if cpp_available else 'disabled')
        
        # ニューラルネットワーク
        self._add_radio_pair(left_col, 'NN', self.use_nn, [('有効', True), ('無効', False)], 
                            state='normal' if nn_available else 'disabled')
        
        # 探索モード
        self.search_mode_box = self._add_radio_group(left_col, '探索アルゴリズム', self.search_mode, 
                                                   [('Hybrid', 'hybrid'), ('αβのみ', 'ab_only'), ('MCTSのみ', 'mcts_only')])
        
        # 最大思考時間 (1sの設定を0.5から1.0に修正)
        self._add_radio_group(left_col, '持ち時間設定', self.time_limit, 
                            [('1秒(不安定)', '1.0'), ('3秒', '3.0'), ('5秒', '5.0'), ('10秒', '10.0'), ('30秒', '30.0')])
        
        # プレイヤー
        self._add_radio_pair(left_col, 'プレイヤーの色', self.player_color, [('黒 (先手)', 'black'), ('白 (後手)', 'white')])

        # --- 右列項目 ---
        # 最適化レベル (重複項目を整理)
        self._add_radio_group(right_col, '枝刈り', self.pruning_level, 
                            [('Multi-Cut,MCTS追加', 'extreme'), ('MCTS枝刈り追加', 'aggressive'), ('標準', 'standard'), ('なし', 'none')])
        
        # TT / Pondering
        self._add_radio_pair(right_col, 'TT再利用', self.use_tt, [('有効', True), ('無効', False)])
        self._add_radio_pair(right_col, 'Pondering', self.use_pondering, [('有効', True), ('無効', False)])
        
        # 定石
        self._add_radio_group(right_col, '定石 (Opening Book)', self.book_source, 
                            [('Egaroucid', 'egaroucid'), ('JSON', 'json'), ('なし', 'none')])
        
        self.book_usage_scale = self._add_scale_group(right_col, 'JSON定石採用率 (%)', self.book_usage, 0, 100)
        self.exact_scale = self._add_scale_group(right_col, '読み切り開始 (空きマス数)', self.exact_threshold, 18, 32)

        # エラーメッセージ表示エリア
        self.error_frame = tk.Frame(main_frame, bg='#f8fafc')
        self.error_frame.pack(fill='x', pady=(10, 0))
        
        if not cpp_available:
            tk.Label(self.error_frame, text='C++エンジン未ロード: Python実装を使用します', fg='#dc2626', bg='#f8fafc', font=('Yu Gothic UI', 9)).pack(anchor='w')
        if not nn_available:
            msg = f'NN利用不可: {nn_reason}' if nn_reason else 'NN利用不可: αβのみで動作します'
            tk.Label(self.error_frame, text=msg, fg='#dc2626', bg='#f8fafc', font=('Yu Gothic UI', 9)).pack(anchor='w')

        # 下部ボタンエリア
        button_frame = tk.Frame(main_frame, bg='#f8fafc')
        button_frame.pack(fill='x', pady=(20, 0))
        
        start_btn = tk.Button(button_frame, text='START GAME', command=self.on_ok, 
                             bg='#2563eb', fg='white', font=('Yu Gothic UI', 11, 'bold'), 
                             padx=30, pady=8, relief='flat', cursor='hand2')
        start_btn.pack(side='right')
        
        # イベントトレース
        self.use_nn.trace_add('write', self._on_settings_changed)
        self.search_mode.trace_add('write', self._on_settings_changed)
        self.book_source.trace_add('write', self._on_settings_changed)
        self._on_settings_changed()

        # ダイアログ表示
        self.top.protocol('WM_DELETE_WINDOW', self.on_ok)
        self.top.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.top.winfo_width()) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.top.winfo_height()) // 2)
        self.top.geometry(f'+{x}+{y}')
        self.top.wait_window()

    def _add_radio_pair(self, parent, label, variable, options, state='normal'):
        box = tk.LabelFrame(parent, text=label, padx=10, pady=8, bg='#f8fafc', font=('Yu Gothic UI', 9, 'bold'))
        box.pack(fill='x', pady=(0, 10))
        for text, value in options:
            tk.Radiobutton(box, text=text, value=value, variable=variable, state=state, 
                          bg='#f8fafc', activebackground='#f8fafc').pack(side='left', padx=(0, 15))
        return box

    def _add_radio_group(self, parent, label, variable, options, state='normal'):
        box = tk.LabelFrame(parent, text=label, padx=10, pady=8, bg='#f8fafc', font=('Yu Gothic UI', 9, 'bold'))
        box.pack(fill='x', pady=(0, 10))
        container = tk.Frame(box, bg='#f8fafc')
        container.pack(fill='x')
        for i, (text, value) in enumerate(options):
            rb = tk.Radiobutton(container, text=text, value=value, variable=variable, state=state, 
                               bg='#f8fafc', activebackground='#f8fafc')
            rb.grid(row=i // 2, column=i % 2, sticky='w', padx=(0, 10), pady=2)
        return box

    def _add_scale_group(self, parent, label, variable, min_v, max_v):
        box = tk.LabelFrame(parent, text=label, padx=10, pady=5, bg='#f8fafc', font=('Yu Gothic UI', 9, 'bold'))
        box.pack(fill='x', pady=(0, 10))
        scale = tk.Scale(box, from_=min_v, to=max_v, orient='horizontal', variable=variable, 
                        bg='#f8fafc', highlightthickness=0, length=180)
        scale.pack(fill='x')
        return scale

    def _on_settings_changed(self, *_):
        nn_on = self.use_nn.get()
        mode = self.search_mode.get()
        
        # MCTSモードの制御
        if not nn_on and mode == 'mcts_only':
            self.search_mode.set('ab_only')
            
        # 探索アルゴリズムのRadiobutton状態更新
        for child in self.search_mode_box.winfo_children():
            if isinstance(child, tk.Frame):
                for rb in child.winfo_children():
                    if isinstance(rb, tk.Radiobutton):
                        if rb['text'] == 'αβのみ':
                            rb.config(state='normal')
                        else:
                            rb.config(state='normal' if nn_on else 'disabled')

        # 定石採用率スケールの制御
        book_source = self.book_source.get()
        self.book_usage_scale.config(state='normal' if book_source == 'json' else 'disabled')

    def on_ok(self):
        nn_enabled = self.use_nn.get()
        mode = self.search_mode.get()
        time_limit = float(self.time_limit.get())
        pruning = self.pruning_level.get()
        
        # 内部変数へのマッピング
        res = {
            'use_cpp': self.use_cpp.get() and self.cpp_available,
            'use_nn': nn_enabled,
            'use_mcts': nn_enabled and mode != 'ab_only',
            'search_mode': mode,
            'use_mcts_only': nn_enabled and mode == 'mcts_only',
            'mcts_influence': 0 if mode == 'ab_only' else 100 if mode == 'mcts_only' else 50,
            'book_source': self.book_source.get(),
            'book_usage': self.book_usage.get() if self.book_source.get() == 'json' else 0,
            'use_book': self.book_source.get() != 'none',
            'use_tt': self.use_tt.get(),
            'time_limit': time_limit,
            'player_color': self.player_color.get(),
            'auto_time': True,
            'use_pondering': self.use_pondering.get(),
            
            # 枝刈り設定の統合マッピング
            'pruning_enabled': pruning != 'none',
            'traditional_pruning_enabled': pruning != 'none',
            'mcts_pruning_enabled': pruning in ('extreme', 'aggressive') and nn_enabled and mode != 'ab_only',
            'multi_cut_enabled': pruning == 'extreme',
            
            'mcts_pruning_time': self.mcts_pruning_time.get(),
            'mcts_pruning_branches': self.mcts_pruning_branches.get(),
            'ab_pruning_time': self.ab_pruning_time.get(),
            'use_pondering': self.use_pondering.get(),
            'exact_threshold': self.exact_threshold.get(),
            'multi_cut_threshold': self.multi_cut_threshold.get(),
            'multi_cut_depth': self.multi_cut_depth.get()
        }
        self.result = res
        self.top.destroy()
