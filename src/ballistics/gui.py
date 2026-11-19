import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ballistics.physics import simulate


class BallisticCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Баллистический калькулятор")
        self.root.geometry("1050x800")

        self.anim = None
        self.current_traj = None
        self.vars = {}

        self.create_controls()
        self.create_results()
        self.create_plot()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_controls(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)

        self.params = [
            ("Начальная скорость, м/с", "v0", "50"),
            ("Угол броска, градусы", "angle", "45"),
            ("Начальная высота, м", "h0", "0"),
            ("g, м/с²", "g", "9.81"),
            ("Масса, кг", "mass", "1.0"),
            ("Коэффициент Cd", "cd", "0.47"),
            ("Площадь сечения, м²", "area", "0.01"),
            ("Плотность воздуха, кг/м³", "rho", "1.225"),
        ]

        for i, (label_text, key, default_value) in enumerate(self.params):
            row = i // 4
            col = (i % 4) * 2

            ttk.Label(control_frame, text=label_text).grid(
                row=row, column=col, sticky="e", padx=5, pady=4
            )

            var = tk.StringVar(value=default_value)
            self.vars[key] = var
            ttk.Entry(control_frame, textvariable=var, width=12).grid(
                row=row, column=col + 1, padx=5, pady=4
            )

        for c in range(8):
            control_frame.columnconfigure(c, weight=1)

        self.drag_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_frame, text="Учитывать сопротивление воздуха", variable=self.drag_var
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=5, pady=8)

        ttk.Button(
            control_frame, text="Рассчитать и анимировать", command=self.run_calculation
        ).grid(row=2, column=4, columnspan=4, sticky="ew", padx=5, pady=8)

    def create_results(self):
        self.results_text = tk.Text(
            self.root,
            height=9,
            state="disabled",
            font=("Consolas", 10),
        )
        self.results_text.pack(fill=tk.X, padx=10, pady=5)

    def create_plot(self):
        self.figure = Figure(figsize=(9, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Траектория полёта")
        self.ax.set_xlabel("x, м")
        self.ax.set_ylabel("y, м")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def get_float(self, key, field_name):
        text = self.vars[key].get().strip().replace(",", ".")
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"Поле «{field_name}» должно быть числом.")

    def validate_inputs(self, values):
        rules = [
            (values["v0"] > 0, "Начальная скорость должна быть больше нуля."),
            (0 <= values["angle"] <= 90, "Угол броска должен быть от 0 до 90 градусов."),
            (values["h0"] >= 0, "Начальная высота не может быть отрицательной."),
            (values["g"] > 0, "Ускорение свободного падения должно быть больше нуля."),
            (values["mass"] > 0, "Масса должна быть больше нуля."),
            (
                values["cd"] >= 0 and values["area"] >= 0 and values["rho"] >= 0,
                "Cd, площадь и плотность воздуха не могут быть отрицательными.",
            ),
        ]
        for ok, message in rules:
            if not ok:
                raise ValueError(message)

    def stop_animation(self):
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None

    def run_calculation(self):
        try:
            values = {}
            for label_text, key, _ in self.params:
                values[key] = self.get_float(key, label_text)
            self.validate_inputs(values)
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return

        use_drag = self.drag_var.get()
        self.stop_animation()

        current = simulate(use_drag=use_drag, **values)
        vacuum = current if not use_drag else simulate(use_drag=False, **values)

        self.draw_plot(vacuum, current, use_drag)
        self.show_results(vacuum, current, use_drag)

    def draw_plot(self, vacuum, current, use_drag):
        self.ax.clear()
        self.ax.set_title("Траектория полёта")
        self.ax.set_xlabel("x, м")
        self.ax.set_ylabel("y, м")
        self.ax.grid(True, alpha=0.3)

        self.ax.plot(
            vacuum["x"], vacuum["y"], linestyle="--", color="gray", label="Без сопротивления"
        )

        if use_drag:
            self.ax.plot(
                current["x"],
                current["y"],
                color="tab:blue",
                alpha=0.35,
                label="С сопротивлением",
            )
        (self.traj_line,) = self.ax.plot([], [], color="tab:red", linewidth=2, label="Анимация")
        (self.point,) = self.ax.plot([], [], marker="o", color="red", markersize=7)

        max_x = max(vacuum["range"], current["range"]) or 1.0
        max_y = max(vacuum["max_height"], current["max_height"]) or 1.0
        self.ax.set_xlim(-0.02 * max_x, 1.02 * max_x)
        self.ax.set_ylim(-0.02 * max_y, 1.08 * max_y)
        self.ax.legend(loc="upper right")
        self.canvas.draw()

        n = len(current["x"])
        step = max(1, n // 200)
        frames = list(range(0, n, step))
        if frames[-1] != n - 1:
            frames.append(n - 1)

        self.current_traj = current

        self.anim = FuncAnimation(
            self.figure,
            self.animate,
            frames=frames,
            interval=30,
            repeat=False,
        )

    def animate(self, frame):
        data = self.current_traj
        i = int(frame)
        self.traj_line.set_data(data["x"][: i + 1], data["y"][: i + 1])
        self.point.set_data([data["x"][i]], [data["y"][i]])

    def show_results(self, vacuum, current, use_drag):
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)

        mode = "с сопротивлением воздуха" if use_drag else "без сопротивления воздуха"
        lines = [
            f"Режим: {mode}",
            "",
            f"Время полёта: {current['time']:.3f} с",
            f"Дальность: {current['range']:.3f} м",
            f"Максимальная высота: {current['max_height']:.3f} м",
            f"Скорость падения: {current['impact_speed']:.3f} м/с",
            f"Угол падения: {current['impact_angle']:.2f}°",
        ]

        if use_drag:
            lines += [
                "",
                "Сравнение с полётом без сопротивления:",
                f"Время полёта: {vacuum['time']:.3f} с",
                f"Дальность: {vacuum['range']:.3f} м",
                f"Максимальная высота: {vacuum['max_height']:.3f} м",
            ]

        self.results_text.insert("1.0", "\n".join(lines))
        self.results_text.config(state="disabled")

    def on_close(self):
        self.stop_animation()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = BallisticCalculatorApp(root)
    root.mainloop()
