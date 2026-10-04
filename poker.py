
import os
import pickle
import pyspiel

from open_spiel.python.algorithms import cfr


# ============================================================
# Configuration
# ============================================================

ITERATIONS = 10_000
SOLVER_FILE = "poker_cfr_solver.pkl"

# Two-player No-Limit Texas Hold'em.
#
# fcpa action abstraction:
#   fold
#   check/call
#   pot-sized bet/raise
#   all-in
#
# This is vastly smaller than the unabstracted full NLHE game.
GAME_PARAMS = (
    "universal_poker("
    "betting=nolimit,"
    "numPlayers=2,"
    "numRounds=4,"
    "blind=100 50,"
    "firstPlayer=2 1 1 1,"
    "numSuits=4,"
    "numRanks=13,"
    "numHoleCards=2,"
    "numBoardCards=0 3 1 1,"
    "stack=20000 20000,"
    "bettingAbstraction=fcpa"
    ")"
)


# ============================================================
# Build OpenSpiel game
# ============================================================

def create_game():
    return pyspiel.load_game(GAME_PARAMS)


# ============================================================
# Train CFR+
# ============================================================

def train_solver(iterations=ITERATIONS):
    print("Creating OpenSpiel poker game...")
    game = create_game()

    print("Game:")
    print(game)

    print()
    print(f"Training CFR+ for {iterations:,} iterations...")
    print("This may take a while.")
    print()

    solver = cfr.CFRPlusSolver(game)

    for i in range(iterations):
        solver.evaluate_and_update_policy()

        if (i + 1) % 100 == 0:
            print(f"\rIteration {i + 1:,}/{iterations:,}", end="")

    print("\n")
    print("Training finished.")

    # Save the solver so that we don't have to retrain every time.
    with open(SOLVER_FILE, "wb") as f:
        pickle.dump(solver, f)

    print(f"Solver saved to {SOLVER_FILE}")

    return solver


# ============================================================
# Load existing solver
# ============================================================

def load_solver():
    if not os.path.exists(SOLVER_FILE):
        return None

    print(f"Loading solver from {SOLVER_FILE}...")

    with open(SOLVER_FILE, "rb") as f:
        solver = pickle.load(f)

    print("Solver loaded.")

    return solver


# ============================================================
# Display strategy for an information state
# ============================================================

def show_strategy(solver, state):
    """
    Display the CFR+ average strategy at the current
    information state.
    """

    player = state.current_player()

    if player == pyspiel.PlayerId.TERMINAL:
        print("The state is terminal.")
        return

    if player == pyspiel.PlayerId.CHANCE:
        print("The state is a chance node.")
        return

    info_state = state.information_state_string(player)

    print()
    print("Information state:")
    print(info_state)
    print()

    policy = solver.average_policy()

    if info_state not in policy:
        print("No policy found for this information state.")
        return

    action_probs = policy[info_state]

    print("CFR+ strategy:")
    print("-" * 40)

    for action, probability in action_probs.items():
        try:
            action_name = state.action_to_string(player, action)
        except Exception:
            action_name = str(action)

        print(
            f"{action_name:20s} "
            f"{probability * 100:8.3f}%"
        )


# ============================================================
# Interactive poker state
# ============================================================

def inspect_initial_game(solver):
    """
    Start at the beginning of a hand and show the game state.

    This is primarily a demonstration of how OpenSpiel
    represents the game. It does not yet force specific
    user-entered cards into the chance nodes.
    """

    game = create_game()
    state = game.new_initial_state()

    print()
    print("=" * 60)
    print("OPEN SPIEL POKER STATE")
    print("=" * 60)

    print(state)

    print()
    print("Current player:", state.current_player())

    print("Is chance node:", state.is_chance_node())
    print("Is terminal:", state.is_terminal())

    if state.is_chance_node():
        outcomes = state.chance_outcomes()

        print()
        print("Number of possible chance outcomes:", len(outcomes))

        print()
        print("The game begins with chance dealing cards.")
        print("We will add an interactive card-selection layer next.")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print(" TWO-PLAYER TEXAS HOLD'EM — OPEN SPIEL CFR+")
    print("=" * 60)

    solver = load_solver()

    if solver is None:
        print()
        print("No trained solver found.")
        print()

        answer = input("Train a new CFR+ solver? [y/n]: ").strip().lower()

        if answer != "y":
            return

        solver = train_solver()

    else:
        print()
        print("Using previously trained CFR+ solver.")

    inspect_initial_game(solver)


if __name__ == "__main__":
    main()
