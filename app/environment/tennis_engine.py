import copy
from datetime import datetime
import json
import sys, os

class TennisMatch:
    def __init__(self, player1: str, player2: str, ownerUsername= None, match_id=None, max_sets=3):
        self.match_id = match_id
        self.player1 = player1
        self.player2 = player2
        self.match_moment = MatchMoment()
        self.ownerUsername = ownerUsername
        self.history_undo = []
        self.history_redo = []
        self.max_sets = max_sets
        self.title = f"{player1} x {player2} : {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    def to_dict(self):
        return {
            'idMatch': self.match_id,
            'title': self.title,
            'player1': self.player1,
            'player2': self.player2,
            'ownerUsername': self.ownerUsername,  
            'moments': [self.match_moment.to_dict()],
            
        }

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, indent=4)

    @classmethod
    def from_dict(cls, data):
        match = cls(data['player1'], data['player2'], match_id=data['idMatch'])
        if 'ownerUsername' in data:
            match.ownerUsername = data['ownerUsername']
        match.match_moment = MatchMoment.from_dict(data['moments'][0])
        match.title = data['title']
        return match

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)

    def start_match(self):
        self.match_moment.current_set = Set()
        self.match_moment.current_game = Game()

    def end_match(self):
        # Perform any necessary cleanup or calculations
        pass

    def point(self, player):
        self.history_undo.append(copy.deepcopy(self.match_moment))
        self.history_redo = []
        if isinstance(self.match_moment.current_game, Tiebreak):
            self.update_tiebreak(player)
        elif isinstance(self.match_moment.current_game, Game):
            self.update_game(player)

    def update_game(self, player):
        if player == self.player1:
            if self.match_moment.current_game.player1_score == '0':
                self.match_moment.current_game.player1_score = '15'
            elif self.match_moment.current_game.player1_score == '15':
                self.match_moment.current_game.player1_score = '30'
            elif self.match_moment.current_game.player1_score == '30':
                self.match_moment.current_game.player1_score = '40'
            elif self.match_moment.current_game.player1_score == '40' and self.match_moment.current_game.player2_score in ['0', '15', '30']:
                self.update_set(player)
            elif self.match_moment.current_game.player1_score == '40' and self.match_moment.current_game.player2_score == '40':
                self.match_moment.current_game.player1_score = 'AD'
            elif self.match_moment.current_game.player1_score == 'AD':
                self.update_set(player)
            elif self.match_moment.current_game.player2_score == 'AD':
                self.match_moment.current_game.player1_score = '40'
                self.match_moment.current_game.player2_score = '40'

        if player == self.player2:
            if self.match_moment.current_game.player2_score == '0':
                self.match_moment.current_game.player2_score = '15'
            elif self.match_moment.current_game.player2_score == '15':
                self.match_moment.current_game.player2_score = '30'
            elif self.match_moment.current_game.player2_score == '30':
                self.match_moment.current_game.player2_score = '40'
            elif self.match_moment.current_game.player2_score == '40' and self.match_moment.current_game.player1_score in ['0', '15', '30']:
                self.update_set(player)
            elif self.match_moment.current_game.player2_score == '40' and self.match_moment.current_game.player1_score == '40':
                self.match_moment.current_game.player2_score = 'AD'
            elif self.match_moment.current_game.player2_score == 'AD':
                self.update_set(player)
            elif self.match_moment.current_game.player1_score == 'AD':
                self.match_moment.current_game.player1_score = '40'
                self.match_moment.current_game.player2_score = '40'

    def update_set(self, player):
        if player == self.player1:
            self.match_moment.current_set.player1_score += 1
            self.match_moment.current_game = Game()
            if self.match_moment.current_set.player1_score >= 6 and self.match_moment.current_set.player1_score - self.match_moment.current_set.player2_score >= 2:
                self.match_moment.sets.append(self.match_moment.current_set)
                self.match_moment.current_set = Set()
                self.match_moment.current_game = Game()
                self.match_moment.match_score_p1 += 1
                return
            elif self.match_moment.current_set.player1_score == 7:
                self.match_moment.sets.append(self.match_moment.current_set)
                self.match_moment.current_set = Set()
                self.match_moment.current_game = Game()
                self.match_moment.match_score_p1 += 1
                return

        if player == self.player2:
            self.match_moment.current_set.player2_score += 1
            self.match_moment.current_game = Game()
            if self.match_moment.current_set.player2_score >= 6 and self.match_moment.current_set.player2_score - self.match_moment.current_set.player1_score >= 2:
                self.match_moment.sets.append(self.match_moment.current_set)
                self.match_moment.current_set = Set()
                self.match_moment.match_score_p2 += 1
                return
            elif self.match_moment.current_set.player2_score == 7:
                self.match_moment.sets.append(self.match_moment.current_set)
                self.match_moment.current_set = Set()
                self.match_moment.current_game = Game()
                self.match_moment.match_score_p2 += 1
                return

        if self.match_moment.current_set.player1_score == 6 and self.match_moment.current_set.player2_score == 6:
            self.match_moment.current_game = Tiebreak()

    def update_tiebreak(self, player):
        if player == self.player1:
            self.match_moment.current_game.player1_score += 1
        if player == self.player2:
            self.match_moment.current_game.player2_score += 1

        if self.match_moment.current_game.player1_score >= self.match_moment.current_game.max_score and self.match_moment.current_game.player1_score - self.match_moment.current_game.player2_score >= self.match_moment.current_game.min_difference:
            self.update_set(self.player1)
        elif self.match_moment.current_game.player2_score >= self.match_moment.current_game.max_score and self.match_moment.current_game.player2_score - self.match_moment.current_game.player1_score >= self.match_moment.current_game.min_difference:
            self.update_set(self.player2)

    def update_history(self):
        self.history_undo.append(self.match_moment)

    def undo(self):
        if len(self.history_undo) > 0:
            self.history_redo.append(copy.deepcopy(self.match_moment))
            self.match_moment = self.history_undo.pop()
            return
        return

    def redo(self):
        if len(self.history_redo) > 0:
            self.history_undo.append(copy.deepcopy(self.match_moment))
            self.match_moment = self.history_redo.pop()
            return
        return

    def relatorio(self):
        #print("Match id: ", self.match_id)
        #print("Player 1: ", self.player1)
        #print("Player 2: ", self.player2)

        for set in range(len(self.match_moment.sets)):
            #print(str(set) + " set : ")
            self.match_moment.sets[set].print_scores()
        #print("Current Set: ")
        self.match_moment.current_set.print_scores()
        #print("Current game: ")
        self.match_moment.current_game.print_scores()

class MatchMoment:
    def __init__(self):
        self.idMatch=None
        self.idMatchMoment=None
        self.sets = []
        self.current_set = None
        self.current_game = None
        self.current_game_p1 = '0'
        self.current_game_p2 = '0'
        self.current_set_p1 = 0
        self.current_set_p2 = 0
        self.match_score_p1 = 0
        self.match_score_p2 = 0

    def to_dict(self):
        return {
            'idMatch': self.idMatch,
            'idMatchMoment': self.idMatchMoment,
            'current_game_p1': self.current_game.player1_score,
            'current_game_p2': self.current_game.player2_score,
            'current_set_p1': self.current_set.player1_score,
            'current_set_p2': self.current_set.player2_score,
            'match_score_p1': self.match_score_p1,
            'match_score_p2': self.match_score_p2,
            'sets': [set.to_dict() for set in self.sets],
        }

    @classmethod
    def from_dict(cls, data):
        moment = cls()
        if 'idMatch' in data:
            moment.idMatch = data['idMatch']
        if 'idMatchMoment' in data:
            moment.idMatchMoment = data['idMatchMoment']

        moment.sets = [Set.from_dict(set) for set in data['sets']]
        moment.current_set = Set.from_dict(data)
        moment.match_score_p1 = int(data['match_score_p1'])
        moment.match_score_p2 = int(data['match_score_p2'])

        if moment.current_set.player1_score == 6 and moment.current_set.player2_score == 6:
            moment.current_game = Tiebreak.from_dict(data)
        else:
            moment.current_game = Game.from_dict(data)

        return moment

class Set:
    def __init__(self):
        self.idMatchMoment=None
        self.idMatchSet = None
        self.player1_score = 0
        self.player2_score = 0

    def print_scores(self):
        print("(Set)Player 1 Score:", self.player1_score)
        print("(Set)Player 2 Score:", self.player2_score)

    def to_dict(self):
        return {
            'idMatchMoment': self.idMatchMoment,
            'idMatchSet': self.idMatchSet,
            'p1': self.player1_score,
            'p2': self.player2_score,
        }

    @classmethod
    def from_dict(cls, data):
        set = cls()
        if 'idMatchMoment' in data:
            set.idMatchMoment = data['idMatchMoment']
        if 'idMatchSet' in data:
            set.idMatchSet = data['idMatchSet']
        if 'p1' in data and 'p2' in data:
            set.player1_score = int(data['p1'])
            set.player2_score = int(data['p2'])
        elif 'current_set_p1' in data and 'current_set_p2' in data:
            set.player1_score = int(data['current_set_p1'])
            set.player2_score = int(data['current_set_p2'])
        return set

class Game:
    def __init__(self):
        self.player1_score = '0'
        self.player2_score = '0'

    def print_scores(self):
        print("(Game)Player 1 Score no game:", self.player1_score)
        print("(Game)Player 2 Score no game:", self.player2_score)

    def to_dict(self):
        return {
            'current_game_p1': self.player1_score,
            'current_game_p2': self.player2_score,
        }

    @classmethod
    def from_dict(cls, data):
        game = cls()
        game.player1_score = data['current_game_p1']
        game.player2_score = data['current_game_p2']
        return game

class Tiebreak:
    def __init__(self, max_score=7, min_difference=2):
        self.id=None
        self.player1_score = 0
        self.player2_score = 0
        self.max_score = max_score
        self.min_difference = min_difference

    def print_scores(self):
        print("(Tiebreak)Player 1 Score:", self.player1_score)
        print("(Tiebreak)Player 2 Score:", self.player2_score)

    def to_dict(self):
        return {
            'current_game_p1': self.player1_score,
            'current_game_p2': self.player2_score,
        }

    @classmethod
    def from_dict(cls, data):
        game = cls()
        game.player1_score = int(data['current_game_p1'])
        game.player2_score = int(data['current_game_p2'])
        return game

if __name__ == "__main__":
    # Example usage
    p = TennisMatch('Davi', 'Gustavo', 13)
    p.start_match()

    for i in range(20):
        p.point('Davi')
    for i in range(24):
        p.point('Gustavo')
    for i in range(6):
        p.point('Davi')
    #p.point('Gustavo')

    p.relatorio()
    print("==========================================")
    print(p.to_json())
    print("==========================================")

    q = TennisMatch.from_json("""
    {
    "idMatch": 7,
        "moments": [
            {"current_game_p1": "40",
            "current_game_p2": "30",
            "current_set_p1": 0, 
            "current_set_p2": 1, 
            "idMatch": 7, 
            "idMatchMoment": 3, 
            "match_score_p1": 1, 
            "match_score_p2": 2, 
            "sets": 
                [
                {"idMatchMoment": 3, 
                "idMatchSet": 9, 
                "p1": 7, "p2": 6}, 
                {"idMatchMoment": 3, 
                "idMatchSet": 10, 
                "p1": 4, "p2": 6}, 
                {"idMatchMoment": 3, 
                "idMatchSet": 11, 
                "p1": 6, 
                "p2": 3},
                {"idMatchMoment": 3, 
                "p1": 6, 
                "p2": 1}
                ]
            }
        ], 
        "ownerUsername": "Rafael", 
        "player1": "Jonas", 
        "player2": "Bob", 
        "title": 
        " Alice vs Boooo"
    }

    """)
    """q.relatorio()
    q.point('Jonas')
    q.relatorio()
    #print(q.to_json())"""