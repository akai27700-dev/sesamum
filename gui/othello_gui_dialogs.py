import tkinter as tk


class StartupSettingsDialog:
    def __init__(self, parent, cpp_available):
        self.result = None
        self.top = tk.Toplevel(parent)
        self.top.title("探索設定")
        self.top.transient(parent)
        self.top.grab_set()
        self.top.resizable(False, False)

        self.cpp_available = cpp_available
        self.use_cpp = tk.StringVar(value="on" if cpp_available else "off")
        self.use_nn = tk.StringVar(value="on")
        self.use_mcts = tk.StringVar(value="on")
        self.use_tt = tk.StringVar(value="on")
        self.mcts_influence = tk.IntVar(value=50)
        self.book_source = tk.StringVar(value="egaroucid")
        self.book_usage = tk.IntVar(value=50)
        self.time_limit = tk.StringVar(value="5")  # デフォルトを5秒に変更
        self.player_color = tk.StringVar(value="black")
        self.use_pondering = tk.BooleanVar(value=True)

        frame = tk.Frame(self.top, padx=16, pady=14)
        frame.pack(fill="both", expand=True)

        self._add_radio_group(frame, "C++探索", self.use_cpp, [("ON", "on"), ("OFF", "off")], state="normal" if cpp_available else "disabled")
        self._add_radio_group(frame, "NN", self.use_nn, [("ON", "on"), ("OFF", "off")])
        self.mcts_influence_scale = self._add_scale_group(frame, "MCTS影響度(0=OFF)", self.mcts_influence, 0, 100)
        self._add_radio_group(frame, "TT再利用", self.use_tt, [("ON", "on"), ("OFF", "off")])
        self._add_radio_group(frame, "常に思考", self.use_pondering, [("ON", "on"), ("OFF", "off")])
        self._add_radio_group(frame, "定石ソース", self.book_source, [("JSON", "json"), ("Egaroucid", "egaroucid")])
        self.book_usage_scale = self._add_scale_group(frame, "定石使用確率(JSONのみ)", self.book_usage, 0, 100)
        self._add_radio_group(frame, "最大思考時間", self.time_limit, [("1秒", "0.5"), ("5秒", "5"), ("10秒", "10"), ("30秒", "30")])
        self._add_radio_group(frame, "プレイヤー", self.player_color, [("黒", "black"), ("白", "white")])
        self.use_nn.trace_add("write", self._toggle_nn_options)
        self.mcts_influence.trace_add("write", self._toggle_nn_options)
        self.book_source.trace_add("write", self._toggle_book_options)
        self._toggle_nn_options()
        self._toggle_book_options()

        if not cpp_available:
            tk.Label(frame, text="C++エンジンが未ロードのため Python 実装で動作します。", anchor="w", fg="#b71c1c").pack(fill="x", pady=(4, 8))

        btns = tk.Frame(frame)
        btns.pack(fill="x", pady=(8, 0))
        tk.Button(btns, text="start", command=self.on_ok, width=10).pack(side="right", padx=(8, 0))

        self.top.protocol("WM_DELETE_WINDOW", self.on_ok)
        self.top.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.top.winfo_width()) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.top.winfo_height()) // 2)
        self.top.geometry(f"+{x}+{y}")
        self.top.wait_window()

    def _add_radio_group(self, parent, label, variable, options, state="normal"):
        box = tk.LabelFrame(parent, text=label, padx=8, pady=6)
        box.pack(fill="x", pady=(0, 8))
        for text, value in options:
            tk.Radiobutton(box, text=text, value=value, variable=variable, state=state).pack(side="left", padx=(0, 10))
        return box

    def _add_scale_group(self, parent, label, variable, min_value, max_value):
        box = tk.LabelFrame(parent, text=label, padx=8, pady=6)
        box.pack(fill="x", pady=(0, 8))
        scale = tk.Scale(box, from_=min_value, to=max_value, orient="horizontal", variable=variable, resolution=1)
        scale.pack(fill="x")
        return scale

    def _add_checkbox(self, parent, label, variable):
        box = tk.LabelFrame(parent, text=label, padx=8, pady=6)
        box.pack(fill="x", pady=(0, 8))
        checkbox = tk.Checkbutton(box, text="有効", variable=variable)
        checkbox.pack(side="left")
        return box

    def _toggle_nn_options(self, *_):
        nn_on = self.use_nn.get() == "on"
        mcts_pct = self.mcts_influence.get()
        self.mcts_influence_scale.config(state="normal" if nn_on else "disabled")

    def _toggle_book_options(self, *_):
        book_json = self.book_source.get() == "json"
        self.book_usage_scale.config(state="normal" if book_json else "disabled")

    def on_ok(self):
        nn_enabled = self.use_nn.get() == "on"
        mcts_pct = self.mcts_influence.get()
        book_usage_pct = self.book_usage.get()
        time_val = self.time_limit.get()
        time_limit = float(time_val)
        self.result = {
            "use_cpp": self.use_cpp.get() == "on" and self.cpp_available,
            "use_nn": nn_enabled,
            "use_mcts": nn_enabled and mcts_pct > 0,
            "mcts_influence": mcts_pct if (nn_enabled and mcts_pct > 0) else 0,
            "book_source": self.book_source.get(),
            "book_usage": book_usage_pct if self.book_source.get() == "json" else 0,
            "use_tt": self.use_tt.get() == "on",
            "time_limit": time_limit,
            "player_color": self.player_color.get(),
            "auto_time": True,  # 常に自動調整モード
            "auto_mode_type": "normal",
            "use_pondering": self.use_pondering.get(),
        }
        self.top.destroy()
