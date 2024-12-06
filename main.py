#################### importy
import pygame, sys
import numpy as np
from Button import Button
from StartStopHandler import StartStopHandler
from Manager import Manager
from my_dataclasses import data_game_class
from GameBoard import GameBoard
from Team_ABO_Engine import TeamABOEngine
from colors import *
from SaveHandler import SaveHandler

#################### User setting for AI
NCOLs = 11 # zde nastavuji pocet sloupců až 11
NROWs = 8 # zde nastavuji pocet řádek až 8, tj max 88 her, 172 agentů

#################### General settings of pygame
WIDTH = 1800
HEIGHT = 1080
FPS = 45
nGAMES = NCOLs * NROWs

#################### dictionary setting for gen AI interface
SETTING = {"RUN": True, "SAVE": False,
           "Engine": TeamABOEngine, # zde si prohodím engine
           "Generations": 7,
           "PaMutation": 0.1
           }

####################  pygame engine start + modules
# pygame
pygame.init()
pygame.mixer.init()

# Nastaveni okna aj.
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic tac toe engine trainer")

# hodiny - FPS CLOCK / heart rate
clock = pygame.time.Clock()

# Kolecke spritů
my_sprites = pygame.sprite.Group()

#################### Graphics part
# pridani tlacitek
handler = StartStopHandler(SETTING)
button = Button(1620, 50, 150, 50, "Play/stop", handler)
my_sprites.add(button)

font = pygame.font.Font(None, 25)

savehandler = SaveHandler(SETTING)
savebutton = Button(1620, 125, 150, 50, "Save", savehandler)
my_sprites.add(savebutton)

#################### GAME start:
game_manager = Manager(nGAMES, SETTING)

# setup boards for testing!!!!!
boards = []
for i in range(NCOLs):
    for j in range(NROWs):
        boards.append(GameBoard(50+(i)*140, 30 + j * 130, screen))

####################################################################################################
#######################################       app cycle
running = True
while running:
    # Logika:
    if SETTING["RUN"]:
        game_manager.update()

    # Fetch data:
    id_game = game_manager.get_id_last_updated_game()
    boards[id_game].set_game(
        data_game_class(board=game_manager.ith_board(id_game),
                        player1name="test1", player2name="test2",
                        player1score=game_manager.ith_get_scores(id_game,1),
                        player2score=game_manager.ith_get_scores(id_game,2),
                        active=game_manager.ith_active(id_game))
                        )

    # FPS kontrola / jeslti bezi dle rychlosti!
    clock.tick(FPS)

    # Event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update sprites (zatim neni vyuzito, ale ok)
    my_sprites.update()

    # Render graphics
    screen.fill(WHITE)
    my_sprites.draw(screen)
    # Draw boards:
    [i.draw() for i in boards]

    # Draw text:
    text = font.render(game_manager.get_info_text()+" / "+str(SETTING["Generations"]), True, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.x = 1625
    text_rect.y = 250
    screen.blit(text, text_rect)

    cstat = game_manager.get_current_stats()
    text_active_games = font.render("Active games: "+str(cstat["active_games"]), True, (0, 0, 0))
    text_rect_active_games = text_active_games.get_rect()
    text_rect_active_games.x = 1625
    text_rect_active_games.y = 300
    screen.blit(text_active_games, text_rect_active_games)

    text_wins_games = font.render("Wins: " + str(cstat["wins"]), True, (0, 0, 0))
    text_rect_wins_games = text_wins_games.get_rect()
    text_rect_wins_games.x = 1625
    text_rect_wins_games.y = 350
    screen.blit(text_wins_games, text_rect_wins_games)

    text_draws_games = font.render("Draws: " + str(cstat["draws"]), True, (0, 0, 0))
    text_rect_draws_games = text_draws_games.get_rect()
    text_rect_draws_games.x = 1625
    text_rect_draws_games.y = 400
    screen.blit(text_draws_games, text_rect_draws_games)

    # previous stat
    pstat=game_manager.get_previous_stats()
    text_wins_games = font.render("P. wins: " + str(pstat["wins"]), True, (0, 0, 0))
    text_rect_wins_games = text_wins_games.get_rect()
    text_rect_wins_games.x = 1625
    text_rect_wins_games.y = 450
    screen.blit(text_wins_games, text_rect_wins_games)

    text_draws_games = font.render("P. draws: " + str(pstat["draws"]), True, (0, 0, 0))
    text_rect_draws_games = text_draws_games.get_rect()
    text_rect_draws_games.x = 1625
    text_rect_draws_games.y = 500
    screen.blit(text_draws_games, text_rect_draws_games)

    if not game_manager.playing:
        running = False

    if SETTING["SAVE"]:
        game_manager.button_save_game()
        SETTING["SAVE"]= False

    pygame.display.flip()

pygame.quit()
