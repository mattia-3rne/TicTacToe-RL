import tkinter as tk
import numpy as np
import random

# Define the game board size
BOARD_SIZE = 3

# Define the possible states for each square
EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'

# Define the possible rewards
REWARD_WIN = 1
REWARD_LOSS = -1
REWARD_DRAW = 0

# Define the Q-learning hyperparameters
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1

class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        master.title("Tic Tac Toe")
        
        # Create the game board grid
        self.board = []
        for i in range(BOARD_SIZE):
            row = []
            for j in range(BOARD_SIZE):
                button = tk.Button(master, text="", width=10, height=3, font=('Helvetica', 20),
                                  command=lambda row=i, col=j: self.make_move(row, col))
                button.grid(row=i, column=j, padx=5, pady=5)
                row.append(button)
            self.board.append(row)
        
        # Create the Q-learning agent
        self.agent = QLearningAgent()
        
        # Start the game
        self.current_player = PLAYER_X
        self.game_over = False
    
    def reset_board(self):
        # Reset the game board
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                self.board[i][j]["text"] = ""
        self.current_player = PLAYER_X
        self.game_over = False
    
    def make_move(self, row, col):
        # Check if the square is empty and the game is not over
        if self.board[row][col]["text"] == "" and not self.game_over:
            # Make the current player's move
            self.board[row][col]["text"] = self.current_player
        
            # Check if the game is over after the current player's move
            if self.check_win(self.current_player):
                self.game_over = True
                self.show_result(self.current_player)
                if self.current_player == PLAYER_X:
                    self.agent.update_q_values(self.get_board_state(), None, REWARD_LOSS)
                else:
                    self.agent.update_q_values(self.get_board_state(), None, REWARD_WIN)
            elif self.check_draw():
                self.game_over = True
                self.show_result("Draw")
                self.agent.update_q_values(self.get_board_state(), None, REWARD_DRAW)
            else:
                # Switch to the other player
                self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
                if self.current_player == PLAYER_O:
                    # Get the agent's move
                    self.agent_move()
                    # Check if the game is over after the agent's move
                    if self.check_win(PLAYER_O):
                        self.game_over = True
                        self.show_result(PLAYER_O)
                        self.agent.update_q_values(self.get_board_state(), None, REWARD_WIN)
                    elif self.check_draw():
                        self.game_over = True
                        self.show_result("Draw")
                        self.agent.update_q_values(self.get_board_state(), None, REWARD_DRAW)
                else:
                    # If it's the human player's turn, no further action needed
                    pass

    
    def agent_move(self):
        # Get available moves
        available_moves = self.get_available_moves()
        # Get the agent's move
        agent_move = self.agent.choose_move(self.get_board_state(), available_moves)
        # Make the agent's move
        self.board[agent_move[0]][agent_move[1]]["text"] = PLAYER_O
    
    def get_board_state(self):
        # Get the current state of the board as a string
        state = ""
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                state += self.board[i][j]["text"]
        return state
    
    def get_available_moves(self):
        # Get a list of available moves on the board
        available_moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]["text"] == "":
                    available_moves.append((i, j))
        return available_moves
    
    def check_win(self, player):
        # Check if the given player has won the game
        # Check rows
        for i in range(BOARD_SIZE):
            if all(self.board[i][j]["text"] == player for j in range(BOARD_SIZE)):
                return True
        # Check columns
        for i in range(BOARD_SIZE):
            if all(self.board[j][i]["text"] == player for j in range(BOARD_SIZE)):
                return True
        # Check diagonals
        if all(self.board[i][i]["text"] == player for i in range(BOARD_SIZE)):
            return True
        if all(self.board[i][BOARD_SIZE-i-1]["text"] == player for i in range(BOARD_SIZE)):
            return True
        return False
    
    def check_draw(self):
        # Check if the game is a draw
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j]["text"] == "":
                    return False
        return True
    
    def show_result(self, winner):
        # Display the winner or draw message
        if winner == "Draw":
            result = "It's a draw!"
        else:
            result = f"{winner} wins!"
        result_label = tk.Label(self.master, text=result, font=('Helvetica', 18))
        result_label.grid(row=BOARD_SIZE, columnspan=BOARD_SIZE, pady=10)

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        
    def choose_move(self, state, available_moves):
        # Choose the best move using the Q-table
        if state not in self.q_table:
            # Initialize Q-values for unseen state
            self.q_table[state] = {move: 0 for move in available_moves}
            return random.choice(available_moves)

        if random.random() < EPSILON:
            # Choose a random move for exploration
            return random.choice(available_moves)
        else:
            # Choose the move with the highest Q-value
            valid_moves = [move for move in available_moves if move in self.q_table[state]]
            if valid_moves:
                return max(valid_moves, key=self.q_table[state].get)
            else:
                # If all moves are unseen, choose randomly
                return random.choice(available_moves)



    
    def update_q_values(self, state, action, reward):
        # Update the Q-values using the Q-learning update rule
        if state not in self.q_table:
            self.q_table[state] = {move: 0 for move in self.get_available_moves(state)}
        
        if action is None:
            # No action taken, so no update needed
            return
        
        # Get the next state and available moves
        next_state = self.get_next_state(state, action)
        next_available_moves = self.get_available_moves(next_state)
        
        # Calculate the Q-value update
        max_next_q = max([self.q_table[next_state][move] for move in next_available_moves], default=0)
        self.q_table[state][action] = (1 - LEARNING_RATE) * self.q_table[state][action] + \
                                      LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q)
    
    def get_available_moves(self, state):
        # Get a list of available moves for the given state
        if len(state) != BOARD_SIZE * BOARD_SIZE:
            return []
        
        available_moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if state[i * BOARD_SIZE + j] == EMPTY:
                    available_moves.append((i, j))
        return available_moves
    
    def get_next_state(self, state, action):
        # Get the next state after taking the given action
        next_state = list(state)
        next_state[action[0] * BOARD_SIZE + action[1]] = PLAYER_O
        return ''.join(next_state)



def train_agent(agent, num_episodes):
    for episode in range(num_episodes):
        if (episode + 1) % 1000 == 0:
            print(f"Training Episode {episode + 1}/{num_episodes}")
        game.reset_board()
        while not game.game_over:
            game.agent_move()
            if game.check_win(PLAYER_O):
                agent.update_q_values(game.get_board_state(), None, REWARD_WIN)
                break
            elif game.check_draw():
                agent.update_q_values(game.get_board_state(), None, REWARD_DRAW)
                break
            game.current_player = PLAYER_X
            game.make_move(*random.choice(game.get_available_moves()))
    print(f"Finished training {num_episodes}/{num_episodes} episodes")



# Create the GUI
root = tk.Tk()
game = TicTacToeGUI(root)

# Train the agent
train_agent(game.agent, num_episodes=10000)

# Start the GUI main loop
root.mainloop()
