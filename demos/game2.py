# Import the pygame module
import pygame

# Import random for random numbers
import random
from asyncio.windows_utils import pipe
from fileinput import filename
import sys
from pathlib import Path
from pyparsing import White

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add kapao/ to path

import numpy as np
import torch
from utils.datasets import LoadImages, LoadWebcam

from demos.game_startup import startup, parse_the_inputs
from val import run_nms, post_process_batch
from utils.torch_utils import select_device

BLACK = (0, 0, 0)
WHITE_color = (255, 255, 255)
RED = (255, 0, 0)
LIGHT_RED = (255, 150, 150 )

GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (150, 150, 255)
GRAY = (200, 200, 200)

# Import pygame.locals for easier access to key coordinates
# Updated to conform to flake8 and black standards
# from pygame.locals import *
from pygame.locals import (
    RLEACCEL,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

# Define constants for the screen width and height
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

def scrollX(screenSurf, offsetX):
    width, height = screenSurf.get_size()
    copySurf = screenSurf.copy()
    screenSurf.blit(copySurf, (offsetX, 0))
    if offsetX < 0:
        screenSurf.blit(copySurf, (width + offsetX, 0), (0, 0, -offsetX, height))
    else:
        screenSurf.blit(copySurf, (0, 0), (width - offsetX, 0, offsetX, height))

# Define the cloud object extending pygame.sprite.Sprite
# Use an image for a better looking sprite
class Balloon(pygame.sprite.Sprite):
    def __init__(self, speed):
        super(Balloon, self).__init__()
        v = np.random.rand()
        if v > 0.6:
            file = "sprite/ballon_very_small.jpg"
        elif v > 0.3:
            file = "sprite/blue4.png"
        else: 
            file = "sprite/blue.jpg"
        self.surf = pygame.image.load(file). convert_alpha()
        # self.surf
        # self.surf = pygame.image.load(file).convert()

        # self.surf.set_colorkey((254, 254, 254))
        # self.surf.set_colorkey((255, 255, 255))

        # self.surf.set_alpha(128)
        print(f"alpha is {self.surf.get_colorkey()}")
        # The starting position is randomly generated
        center = (
            random.randint(20, SCREEN_WIDTH - 20),
            SCREEN_HEIGHT + self.surf.get_size()[1]
            # random.randint(0, SCREEN_HEIGHT),
        )
        self.rect = self.surf.get_rect(center=center)
        # self.rect.
        print(f"new ballon at {center}")
        self.remove = False
        self.at_top = False
        self.speed = speed + speed * ((np.random.normal() - 0.5) / 4)
        self.zigzag = np.random.rand() > 0.75

    # Move the cloud based on a constant speed
    # Remove it when it passes the left edge of the screen
    def update(self):
        if self.remove:
            self.kill()

        self.rect.move_ip(0, -self.speed)
        if self.zigzag:
            mult = 1.5
            # print(self.rect)
            move_right = False
            move_left = False
            if self.rect.left < 30:
                move_right = True
            elif self.rect.left > (SCREEN_WIDTH - 30):
                move_left = True
            else:
                move_right = np.random.rand() > 0.5
                move_left = not move_right

            if move_right:
                self.rect.move_ip(mult * self.speed, 0)
            if move_left:
                self.rect.move_ip(-mult * self.speed, 0)
            # print(f"Ballon left is {self.rect.left}. ml = {move_left} and mr = {move_right}")
        # print(self.rect)
        if self.rect.bottom < 0:
            # self.kill()
            self.at_top = True
            print("ballon at top")

class Background(pygame.sprite.Sprite):
    def __init__(self):
        super(Background, self).__init__()
        file = "sprite/m_clouds.jpg"
        # file = "sprite/m_sandia.jpg"

        self.surf = pygame.image.load(file).convert()
        self.surf.set_colorkey((0, 0, 0), RLEACCEL)
        # The starting position is randomly generated
        # center = ( SCREEN_WIDTH //2,
        #     SCREEN_HEIGHT
        # )
        center = (0,0)
        self.rect =  self.surf.get_rect() #center=center)

class Hand(pygame.sprite.Sprite):
    color = (255, 255, 255)
    score_multi = 1

    def __init__(self):
        super(Hand, self).__init__()
        # self.x = 0
        self.center = (0, 0)
        self.surf = pygame.Surface((10, 10))
        self.surf.fill(self.color)
        self.rect = self.surf.get_rect(center=self.center)

    def set_center(self, x, y):
        self.center = (x, y)
        self.rect = self.surf.get_rect(center=self.center)


class Foot(Hand):
    color = (0, 255, 0)
    score_multi = 2


class SpriteWithText(pygame.sprite.Sprite):
    def __init__(self, x_position, y_position, text="", background_color = None):
        super(SpriteWithText, self).__init__()
        self.font = pygame.font.SysFont(None, 48)
        # self.update_score(0, 30)
        self.text = text
        self.name = ""
        self.background_color = background_color
        self.update()
        self.rect = self.surf.get_rect(center=(x_position, y_position))

    def update_score(self, score, speed):
        self.score = score
        self.speed = speed
        self.text =   f"Score = {self.score}: Speed = {self.speed}"
        print(f"score is {score}. Speed = {speed}")

    def update(self):
        self.surf = self.font.render(self.text, False, BLACK, self.background_color
        ).convert()


def get_body_poses(dataset, data, model, device, args):
    # get an image from the web cam.
    (_, img, im0, _) = next(iter(dataset))
    img = torch.from_numpy(img).to(device)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    # print
    # (f"img shape is {img.shape}")
    if args.webapp == False:
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        out = model(
            img,
            augment=True,
            kp_flip=data["kp_flip"],
            scales=data["scales"],
            flips=data["flips"],
        )[0]
        person_dets, kp_dets = run_nms(data, out)
    else:
        person_dets, kp_dets = get_server_response(img)
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

    # print("kp_dets shape", kp_dets[0].shape) #([10, 40])
    # print("person_dets shape", person_dets[0].shape) # .Size([1, 40])

    _, poses, _, _, _ = post_process_batch(
            data, img, [], [[im0.shape[:2]]], person_dets, kp_dets
        )
    return poses

def draw_pose_set_body_positions(poses, data, screen, line_color, left_hand= None, right_hand=None, left_foot=None, right_foot=None):
    if len(poses) > 0:
        pose = poses[0]
        # for pose in poses:
        for seg in data["segments"].values():
            pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
            pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
            # cv2.line(im0, pt1, pt2, args.color_pose, args.line_thick)
            pygame.draw.line(screen, line_color, pt1, pt2, width=5)
        # pose = poses[0]

        left_wrist, right_wrist = 9, 10
        left_ankle, right_ankle = 15, 16
        if left_hand:left_hand.set_center(int(pose[left_wrist, 0]), int(pose[left_wrist, 1]))
        if right_hand: right_hand.set_center(int(pose[right_wrist, 0]), int(pose[right_wrist, 1]))

        if left_foot: left_foot.set_center(int(pose[left_ankle, 0]), int(pose[left_ankle, 1]))
        if right_foot: right_foot.set_center(int(pose[right_ankle, 0]), int(pose[right_ankle, 1]))

# class SpriteWithText()

# Initialize pygame
pygame.init()

def start_mode():
    # return True to keep playing
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    left_hand = Hand()
    right_hand = Hand()

    body = pygame.sprite.Group()
    body.add(left_hand)
    body.add(right_hand)


    start_game_box = SpriteWithText(SCREEN_WIDTH*3 //4, 30, "Hit to Start Game", LIGHT_BLUE)
    start_game_box.name = "start"
    exit_game_box = SpriteWithText(SCREEN_WIDTH//4, 30, "Hit to Exit Game", LIGHT_RED)
    exit_game_box.name = "end"

    menu_group = pygame.sprite.Group()
    menu_group.add(start_game_box)
    menu_group.add(exit_game_box)
  
    # body tracking stuff
    args, data, model = startup()
    args.pose = True
    device = select_device(args.device, batch_size=1)
    imgsz = 256
    stride = 64
    dataset = LoadWebcam("0", imgsz, stride)

    line_color = (50, 50, 50)
    running = True
    while running:
        screen.fill(WHITE_color)

        poses = get_body_poses(dataset, data, model, device, args)
        draw_pose_set_body_positions(poses, data, screen, line_color, left_hand,right_hand)

        # pygame.sprite.spritecollide()
        for body_part in body:
            menu_item = pygame.sprite.spritecollideany(body_part, menu_group)
            if menu_item:
                if menu_item.name == "start": return True
                if menu_item.name == "end": return False
        
        # Look at every event in the queue
        for event in pygame.event.get():
            # Did the user hit a key?
            if event.type == KEYDOWN:
                # Was it the Escape key? If so, stop the loop
                if event.key == K_ESCAPE:
                    running = False

            # Did the user click the window close button? If so, stop the loop
            elif event.type == QUIT:
                running = False

        for entity in menu_group:
            screen.blit(entity.surf, entity.rect)
        
        for entity in body:
            screen.blit(entity.surf, entity.rect)
        
        clock.tick(10)
        pygame.display.update()


def end_mode(score,highscore):
    # create the display surface object
    sz = 60
    display_surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.Font('freesansbold.ttf', sz)
    text = font.render(f"Score = {score}",True, BLACK, WHITE_color)

    font2 = pygame.font.Font('freesansbold.ttf', sz)
    highscore_text = font2.render(f"Highscore = {highscore}",
         True, BLACK, WHITE_color)
    textRect = text.get_rect()
    textHighScoreRect = highscore_text.get_rect()

    clock = pygame.time.Clock()
    # set the center of the rectangular object.
    textRect.center = (SCREEN_WIDTH // 2, -50+ SCREEN_HEIGHT// 2)
    textHighScoreRect.center = (SCREEN_WIDTH // 2, 50+ SCREEN_HEIGHT// 2)

    for i in range(50):
        display_surface.fill(WHITE_color)

        # copying the text surface object
        # to the display surface object
        # at the center coordinate.
        display_surface.blit(text, textRect)
        display_surface.blit(highscore_text, textHighScoreRect)


        # iterate over the list of Event objects
        # that was returned by pygame.event.get() method.
        for event in pygame.event.get():
            # if event object type is QUIT
            # then quitting the pygame
            # and program both.
            if event.type == pygame.QUIT:
                return                  
            # Did the user hit a key?
            if event.type == KEYDOWN:
                # Was it the Escape key? If so, stop the loop
                if event.key == K_ESCAPE:
                    return

            # Draws the surface object to the screen.
        pygame.display.update()
        clock.tick(15)



def game_mode():
    # sound
    pygame.mixer.init()
    # collision_sound = pygame.mixer.Sound("sound/pygame-a-primer_Collision.oga")
    collision_sound = pygame.mixer.Sound("sound/pop.wav")

    collision_sound.set_volume(1.0)

    # Setup the clock for a decent framerate
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    # # Create custom events for adding a new enemy and cloud
    # ADDENEMY = pygame.USEREVENT + 1
    # pygame.time.set_timer(ADDENEMY, 250)
    ADDBALLON = pygame.USEREVENT + 2
    new_ballon_rate = 5000
    pygame.time.set_timer(ADDBALLON, new_ballon_rate)

    # # Create our 'player'
    # player = Player()
    left_hand = Hand()
    right_hand = Hand()
    right_foot = Foot()
    left_foot = Foot()


    # # Create groups to hold enemy sprites, cloud sprites, and all sprites
    # # - ballons is used for position updates
    # # - all_sprites isused for rendering
    ballons = pygame.sprite.Group()
    body = pygame.sprite.Group()
    body.add(left_hand)
    body.add(right_hand)
    body.add(right_foot)
    body.add(left_foot)

    score_box = SpriteWithText(int(SCREEN_WIDTH/2), 20, "Score = 0: Speed = 1", background_color=WHITE_color)

    all_sprites = pygame.sprite.Group()
    all_sprites.add(left_hand)
    all_sprites.add(right_hand)
    all_sprites.add(score_box)
    all_sprites.add(right_foot)
    all_sprites.add(left_foot)

    # Variable to keep our main loop running
    running = True
    imgsz = 256
    stride = 64
    dataset = LoadWebcam("0", imgsz, stride)

    args, data, model = startup()
    args.pose = True
    device = select_device(args.device, batch_size=1)

    line_color = (255, 0, 0)
    score = 0

    # Our main loop
    counter = 0
    speed = 3
    speed_increment = 1
    score_box.update_score(score, speed)
    background_change_sign = -1
    sky_color = (135, 206, 250)
    back_ground_sprite = Background()
    while running:
        counter += 1
        print(f"Frame {counter}")

        poses = get_body_poses(dataset, data, model, device, args)
        
        # Look at every event in the queue
        for event in pygame.event.get():
            # Did the user hit a key?
            if event.type == KEYDOWN:
                # Was it the Escape key? If so, stop the loop
                if event.key == K_ESCAPE:
                    running = False

            # Did the user click the window close button? If so, stop the loop
            elif event.type == QUIT:
                running = False

            # Should we add a new ballon?
            elif event.type == ADDBALLON:
                # Create the new cloud, and add it to our sprite groups
               
                new_ballon = Balloon(speed)
                ballons.add(new_ballon)
                all_sprites.add(new_ballon)

        # # Fill the screen with sky blue
        # sky_color = ()
        # screen.fill((135, 206, 250))
        # sky_color = (sky_color[0], sky_color[1] + background_change_sign,  sky_color[2])
        # if sky_color[1]>254:
        #     background_change_sign =-1
        # if sky_color[1]<1:
        #     background_change_sign =1
        screen.fill(sky_color)
        screen.blit(back_ground_sprite.surf, back_ground_sprite.rect)
        # back_ground_sprite.surf.scroll(1,0)
        scrollX(back_ground_sprite.surf, speed // 3)

        if args.pose:            
            draw_pose_set_body_positions(poses, data, screen, line_color, left_hand, right_hand, left_foot, right_foot)

        # print(f"poses shape {poses[0].shape}") 17,3
        # # Check if any enemies have collided with the player
        for body_part in body:
            hit_ballon = pygame.sprite.spritecollideany(body_part, ballons)
            if hit_ballon:
                hit_ballon.remove = True
                score += 100 * body_part.score_multi
                score_box.update_score(score, speed)
                score_box.update()
                collision_sound.play()

        # # Update the position of our enemies and ballons
        ballons.update()

        for b in ballons:
            if b.at_top:
                running = False
                print(f"Final score is {score}")

        # Draw all our sprites
        for entity in all_sprites:
            screen.blit(entity.surf, entity.rect)

        if counter % 100 == 99:
            speed += speed_increment
            new_ballon_rate -= 350
            pygame.time.set_timer(ADDBALLON, new_ballon_rate)
            print(f"Making harder. New speed is {speed}")

        # random new ballons on harder levels.
        if score > 1500:
            ratio = score / 5e5
            if np.random.rand() < ratio:
                num_ballons = max(1+ (score // 3000), 1)
                print(f"Extra ballon ratio = {ratio}, num balloons = {num_ballons}")
                for i in range(num_ballons):
                    new_ballon = Balloon(speed)
                    ballons.add(new_ballon)
                    all_sprites.add(new_ballon)

        # Flip everything to the display
        pygame.display.flip()

        # Ensure we maintain a 10 frames per second rate
        clock.tick(10)



    # At this point, we're done, so we can stop and quit the mixer
    # pygame.mixer.music.stop()
    return score

def save_score(score):
    file = Path("score.txt")
    if file.exists():
        with open(file, 'r') as f:
            highscore = int(f.read())
    else: highscore = 0
    if score> highscore:
        highscore = score
        with open(file, 'w') as f:
            f.write(str(score))
    return highscore

args = parse_the_inputs()
debug = args.debug
if debug == False:
    play = start_mode()
    while play:
        score = game_mode()
        highscore = save_score(score)
        end_mode(score, highscore)
        play = start_mode()
else:
    score = game_mode()


pygame.mixer.quit()
