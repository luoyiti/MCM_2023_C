import tkinter as tk
from tkinter import messagebox, ttk
import random
from collections import Counter
from wordfreq import top_n_list
import threading
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- Logic Functions ---

def get_feedback(guess, true_word):
    feedback = ['gray'] * len(guess)
    answer_counts = Counter(true_word)
    
    # Green pass
    for i, (g, t) in enumerate(zip(guess, true_word)):
        if g == t:
            feedback[i] = 'green'
            answer_counts[g] -= 1
    
    # Yellow pass
    for i, g in enumerate(guess):
        if feedback[i] == 'gray' and answer_counts[g] > 0:
            feedback[i] = 'yellow'
            answer_counts[g] -= 1
    
    return {i: feedback[i] for i in range(len(guess))}

def fit_green(accept_word, words_now):
    return [
        word
        for word in words_now
        if all(
            accept_word[i] == '?' or word[i] == accept_word[i]
            for i in range(len(accept_word))
        )
    ]

def fit_yellow(yellow_info, words_now):
    for letter, forbidden_positions in yellow_info.items():
        words_now = [w for w in words_now if letter in w]
        words_now = [w for w in words_now if all(w[pos] != letter for pos in forbidden_positions)]
    return words_now

def fit_gray(gray_letters, yellow_letters, green_letters, words_now):
    pure_gray = gray_letters - yellow_letters - green_letters
    return [w for w in words_now if all(letter not in w for letter in pure_gray)]

def run_simulation_once(target_word, words_pool):
    """
    Runs a single simulation of the game using the frequency priority rule.
    Returns: (tries, guesses_list)
    """
    yellow_info = {}
    gray_letters = set()
    yellow_letters = set()
    green_letters = set()
    accept_word = ['?'] * 5
    words_now = list(words_pool)
    random_pick_prob = [0.2, 0.3, 0.5]
    
    guesses = []
    
    # First guess
    if not words_now:
        return 7, guesses
        
    top_k = max(1, int(len(words_now) * random.choice(random_pick_prob)))
    guessed_word = random.choice(words_now[:top_k])
    guesses.append(guessed_word)
    
    for try_num in range(1, 8):
        if guessed_word == target_word:
            return try_num, guesses
        
        if try_num == 7:
            return 7, guesses
            
        feedback = get_feedback(guessed_word, target_word)
        
        # Update constraints
        for pos, color in feedback.items():
            guess_char = guessed_word[pos]
            if color == 'green':
                accept_word[pos] = guess_char
                green_letters.add(guess_char)
            elif color == 'yellow':
                yellow_letters.add(guess_char)
                if guess_char not in yellow_info:
                    yellow_info[guess_char] = []
                yellow_info[guess_char].append(pos)
            elif color == 'gray':
                gray_letters.add(guess_char)
                
        words_now = fit_green(accept_word, words_now)
        words_now = fit_yellow(yellow_info, words_now)
        words_now = fit_gray(gray_letters, yellow_letters, green_letters, words_now)
        
        if guessed_word in words_now:
            words_now.remove(guessed_word)
            
        if not words_now:
            return 7, guesses
            
        top_k = max(1, int(len(words_now) * random.choice(random_pick_prob)))
        guessed_word = random.choice(words_now[:top_k])
        guesses.append(guessed_word)
        
    return 7, guesses

# --- GUI Classes ---

class StatsWindow:
    def __init__(self, master):
        self.top = tk.Toplevel(master)
        self.top.title("Simulation Statistics")
        self.top.geometry("600x400")
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Distribution of Tries")
        self.ax.set_xlabel("Number of Tries")
        self.ax.set_ylabel("Frequency")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.counts = {i: 0 for i in range(1, 8)}
        self.total_runs = 0
        self.update_plot()

    def update_data(self, result):
        self.counts[result] += 1
        self.total_runs += 1
        
    def update_plot(self):
        self.ax.clear()
        x = list(self.counts.keys())
        y = list(self.counts.values())
        colors = ['#6aaa64' if i < 7 else '#787c7e' for i in x]
        
        bars = self.ax.bar(x, y, color=colors)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels([str(i) if i < 7 else "Fail" for i in x])
        self.ax.set_title(f"Distribution of Tries (Total: {self.total_runs})")
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            self.ax.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
                            
        self.canvas.draw()

class WordleSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Wordle Simulation (Frequency Priority)")
        
        # Load words
        self.status_var = tk.StringVar()
        self.status_var.set("Loading words...")
        self.root.update()
        
        self.load_words()
        
        self.status_var.set(f"Loaded {len(self.words_5)} words.")
        
        # Game State
        self.target_word = ""
        self.current_try = 0
        self.words_now = []
        self.yellow_info = {}
        self.gray_letters = set()
        self.yellow_letters = set()
        self.green_letters = set()
        self.accept_word = ['?'] * 5
        self.game_over = False
        self.auto_running = False
        self.batch_running = False
        
        # UI Setup
        self.setup_ui()
        self.reset_game()

    def load_words(self):
        try:
            top5000 = top_n_list("en", 500000)
            VOCAB_SIZE = 20000
            self.words_5 = [w for w in top5000 if len(w) == 5 and "'" not in w and w.isalpha() and w.isascii()][:VOCAB_SIZE]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load words: {e}")
            self.words_5 = []

    def setup_ui(self):
        # Main Container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left Side: Single Game
        left_frame = tk.Frame(main_frame, padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right Side: Batch Simulation
        right_frame = tk.Frame(main_frame, padx=10, pady=10, relief=tk.GROOVE, borderwidth=1)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # --- Left Side ---
        
        # Control Panel
        control_frame = tk.Frame(left_frame, pady=10)
        control_frame.pack()
        
        tk.Label(control_frame, text="Target Word:").pack(side=tk.LEFT)
        self.target_entry = tk.Entry(control_frame, width=10)
        self.target_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Set", command=self.set_target).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Random", command=self.set_random_target).pack(side=tk.LEFT, padx=5)
        
        # Grid
        self.grid_frame = tk.Frame(left_frame, padx=20, pady=20)
        self.grid_frame.pack()
        
        self.cells = []
        for row in range(6):
            row_cells = []
            for col in range(5):
                lbl = tk.Label(self.grid_frame, text="", width=4, height=2, font=("Helvetica", 20, "bold"),
                               relief="solid", borderwidth=1, bg="white")
                lbl.grid(row=row, column=col, padx=2, pady=2)
                row_cells.append(lbl)
            self.cells.append(row_cells)
            
        # Action Buttons
        action_frame = tk.Frame(left_frame, pady=10)
        action_frame.pack()
        
        self.btn_step = tk.Button(action_frame, text="Next Step", command=self.step_simulation, state=tk.DISABLED)
        self.btn_step.pack(side=tk.LEFT, padx=5)
        
        self.btn_auto = tk.Button(action_frame, text="Auto Run", command=self.toggle_auto_run, state=tk.DISABLED)
        self.btn_auto.pack(side=tk.LEFT, padx=5)
        
        tk.Button(action_frame, text="Reset", command=self.reset_game).pack(side=tk.LEFT, padx=5)
        
        # Info
        self.info_label = tk.Label(left_frame, textvariable=self.status_var, pady=10)
        self.info_label.pack()
        
        # --- Right Side (Batch) ---
        
        tk.Label(right_frame, text="Batch Simulation", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        tk.Label(right_frame, text="Number of Simulations:").pack(anchor=tk.W)
        self.batch_count_var = tk.IntVar(value=100)
        tk.Entry(right_frame, textvariable=self.batch_count_var).pack(fill=tk.X, pady=5)
        
        tk.Label(right_frame, text="Speed (Delay ms):").pack(anchor=tk.W)
        self.speed_var = tk.IntVar(value=10)
        tk.Scale(right_frame, from_=0, to=500, orient=tk.HORIZONTAL, variable=self.speed_var).pack(fill=tk.X)
        
        self.btn_batch = tk.Button(right_frame, text="Start Batch", command=self.start_batch_simulation, bg="#6aaa64", fg="black")
        self.btn_batch.pack(pady=20, fill=tk.X)
        
        self.batch_progress = ttk.Progressbar(right_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.batch_progress.pack(fill=tk.X, pady=10)

    def set_target(self):
        word = self.target_entry.get().lower().strip()
        if len(word) != 5 or not word.isalpha():
            messagebox.showerror("Error", "Please enter a valid 5-letter word.")
            return
        self.start_game(word)

    def set_random_target(self):
        if not self.words_5:
            return
        word = random.choice(self.words_5)
        self.target_entry.delete(0, tk.END)
        self.target_entry.insert(0, word)
        self.start_game(word)

    def start_game(self, word):
        self.target_word = word
        self.reset_state()
        self.btn_step.config(state=tk.NORMAL)
        self.btn_auto.config(state=tk.NORMAL)
        self.btn_batch.config(state=tk.NORMAL)
        self.status_var.set(f"Target: {word} | Candidates: {len(self.words_now)}")
        self.clear_grid()

    def reset_game(self):
        self.target_entry.delete(0, tk.END)
        self.btn_step.config(state=tk.DISABLED)
        self.btn_auto.config(state=tk.DISABLED)
        self.auto_running = False
        self.btn_auto.config(text="Auto Run")
        self.reset_state()
        self.clear_grid()
        self.status_var.set("Ready. Set a target word to start.")

    def reset_state(self):
        self.current_try = 0
        self.words_now = list(self.words_5)
        self.yellow_info = {}
        self.gray_letters = set()
        self.yellow_letters = set()
        self.green_letters = set()
        self.accept_word = ['?'] * 5
        self.game_over = False

    def clear_grid(self):
        for row in range(6):
            for col in range(5):
                self.cells[row][col].config(text="", bg="white", fg="black")

    def step_simulation(self):
        if self.game_over or self.current_try >= 6:
            return

        # Logic from optimal_rule_according_to_frequency
        random_pick_prob = [0.2, 0.3, 0.5]
        
        if not self.words_now:
            self.status_var.set("No candidates left! Simulation failed.")
            self.game_over = True
            self.stop_auto_run()
            return

        # Pick guess
        top_k = max(1, int(len(self.words_now) * random.choice(random_pick_prob)))
        guessed_word = random.choice(self.words_now[:top_k])
        
        # Display guess
        for i, char in enumerate(guessed_word):
            self.cells[self.current_try][i].config(text=char.upper())
        
        # Check win
        if guessed_word == self.target_word:
            self.color_grid(self.current_try, guessed_word, {i: 'green' for i in range(5)})
            self.status_var.set(f"Solved in {self.current_try + 1} tries!")
            self.game_over = True
            self.stop_auto_run()
            return

        # Get feedback
        feedback = get_feedback(guessed_word, self.target_word)
        self.color_grid(self.current_try, guessed_word, feedback)
        
        # Update constraints
        for pos, color in feedback.items():
            guess_char = guessed_word[pos]
            if color == 'green':
                self.accept_word[pos] = guess_char
                self.green_letters.add(guess_char)
            elif color == 'yellow':
                self.yellow_letters.add(guess_char)
                if guess_char not in self.yellow_info:
                    self.yellow_info[guess_char] = []
                self.yellow_info[guess_char].append(pos)
            elif color == 'gray':
                self.gray_letters.add(guess_char)
        
        # Filter candidates
        self.words_now = fit_green(self.accept_word, self.words_now)
        self.words_now = fit_yellow(self.yellow_info, self.words_now)
        self.words_now = fit_gray(self.gray_letters, self.yellow_letters, self.green_letters, self.words_now)
        
        if guessed_word in self.words_now:
            self.words_now.remove(guessed_word)
            
        self.current_try += 1
        self.status_var.set(f"Candidates left: {len(self.words_now)}")
        
        if self.current_try >= 6 and not self.game_over:
            self.status_var.set("Failed to solve in 6 tries.")
            self.game_over = True
            self.stop_auto_run()

    def color_grid(self, row, word, feedback):
        colors = {'green': '#6aaa64', 'yellow': '#c9b458', 'gray': '#787c7e'}
        for i in range(5):
            color_name = feedback[i]
            self.cells[row][i].config(bg=colors[color_name], fg="white")

    def toggle_auto_run(self):
        if self.auto_running:
            self.stop_auto_run()
        else:
            self.auto_running = True
            self.btn_auto.config(text="Stop Auto")
            self.run_auto_step()

    def stop_auto_run(self):
        self.auto_running = False
        self.btn_auto.config(text="Auto Run")

    def run_auto_step(self):
        if self.auto_running and not self.game_over:
            self.step_simulation()
            self.root.after(500, self.run_auto_step)

    # --- Batch Simulation ---

    def start_batch_simulation(self):
        if not self.target_word:
            messagebox.showwarning("Warning", "Please set a target word first.")
            return
            
        if self.batch_running:
            return

        count = self.batch_count_var.get()
        if count <= 0:
            return

        self.batch_running = True
        self.btn_batch.config(state=tk.DISABLED)
        self.batch_progress["maximum"] = count
        self.batch_progress["value"] = 0
        
        # Open Stats Window
        self.stats_window = StatsWindow(self.root)
        
        # Start Thread
        threading.Thread(target=self.run_batch_loop, args=(count,), daemon=True).start()

    def run_batch_loop(self, count):
        delay = self.speed_var.get() / 1000.0
        
        for i in range(count):
            if not self.batch_running:
                break
                
            tries, guesses = run_simulation_once(self.target_word, self.words_5)
            
            # Update Stats Window (Thread-safe call)
            self.root.after(0, self.stats_window.update_data, tries)
            self.root.after(0, self.stats_window.update_plot)
            
            # Update Progress
            self.root.after(0, lambda v=i+1: self.batch_progress.configure(value=v))
            
            # Optional: Show last game on grid if slow enough
            if delay > 0.05:
                self.root.after(0, self.display_game_result, guesses)
            
            if delay > 0:
                time.sleep(delay)
                
        self.batch_running = False
        self.root.after(0, lambda: self.btn_batch.config(state=tk.NORMAL))

    def display_game_result(self, guesses):
        self.clear_grid()
        for row, guess in enumerate(guesses):
            if row >= 6: break
            for col, char in enumerate(guess):
                self.cells[row][col].config(text=char.upper())
            
            feedback = get_feedback(guess, self.target_word)
            self.color_grid(row, guess, feedback)

if __name__ == "__main__":
    root = tk.Tk()
    app = WordleSimulatorGUI(root)
    root.mainloop()
