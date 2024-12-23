from bgbench.nim_game import NimGame

def main():
    print("Welcome to bgbench!")
    game = NimGame()
    print(game.get_rules_explanation())

if __name__ == "__main__":
    main()
