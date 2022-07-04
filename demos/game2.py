# Import the pygame module
import pygame
# from pygame.locals import *

# Import random for random numbers
import random

from asyncio.windows_utils import pipe
from fileinput import filename
import sys
from pathlib import Path

from demos.webserver_interface import get_server_response
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add kapao/ to path

import numpy as np
import torch
import time

import cv2
from utils.datasets import LoadImages, LoadWebcam

from demos.game_startup import startup
from val import run_nms, post_process_batch
from utils.torch_utils import select_device

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
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


# Define the cloud object extending pygame.sprite.Sprite
# Use an image for a better looking sprite
class Ballon(pygame.sprite.Sprite):
    def __init__(self, speed):
        super(Ballon, self).__init__()
        if np.random.rand()>0.5:
            file = "sprite/ballon_very_small.jpg"
        else:
            file = "sprite/blue_small.jpg"
        self.surf = pygame.image.load(file).convert()
        self.surf.set_colorkey((0, 0, 0), RLEACCEL)
        # The starting position is randomly generated
        center=(
                random.randint( 20, SCREEN_WIDTH - 20),
                SCREEN_HEIGHT+20
                # random.randint(0, SCREEN_HEIGHT),
            )
        self.rect = self.surf.get_rect(center = center)
        print(f"new ballon at {center}")
        self.remove = False
        self.at_top = False
        self.speed = speed + speed*((np.random.rand()-0.5)/5)
        self.zigzag = np.random.rand()>0.75

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
            if self.rect.left<30: move_right = True
            elif self.rect.left>(SCREEN_WIDTH-30): move_left = True               
            else:
                move_right = np.random.rand()>0
                move_left = not move_right
                
            if move_right: self.rect.move_ip(mult*self.speed, 0)
            if move_left:  self.rect.move_ip(-mult*self.speed, 0)
            # print(f"Ballon left is {self.rect.left}. ml = {move_left} and mr = {move_right}")
        # print(self.rect)
        if self.rect.bottom < 0:
            # self.kill()
            self.at_top = True
            print("ballon at top")

class Hand(pygame.sprite.Sprite):
    color = (255, 255, 255)
    score_multi = 1

    def __init__(self):
        super(Hand, self).__init__()
        # self.x = 0
        self.center = (0,0)
        self.surf = pygame.Surface((10, 10))
        self.surf.fill(self.color)
        self.rect = self.surf.get_rect(center = self.center)

    def set_center(self, x,y):
        self.center = (x,y)
        self.rect = self.surf.get_rect(center = self.center)
        
class Foot(Hand):
    color = (255, 0, 0)
    score_multi = 2

class Score_Box(pygame.sprite.Sprite):
    def __init__(self):
        super(Score_Box, self).__init__()
        self.font = pygame.font.SysFont(None, 48)
        self.update_score(0,30)
        self.update()
        self.speed = 0
        self.score = 0
        self.rect = self.surf.get_rect(center = (int(SCREEN_WIDTH/2),20))

    def update_score (self, score, speed):
        self.score = score
        self.speed = speed
        print(f"score is {score}. Speed = {speed}")
       
    def update(self):
        self.surf = self.font.render(f"Score = {self.score}: Speed = {self.speed}", False,BLACK).convert()



# Setup for sounds, defaults are good
# pygame.mixer.init()

# Initialize pygame
pygame.init()

# sound
pygame.mixer.init()
# collision_sound = pygame.mixer.Sound("sound/pygame-a-primer_Collision.oga")
collision_sound = pygame.mixer.Sound("sound/pop.wav")

collision_sound.set_volume(0.5)

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
# # - enemies is used for collision detection and position updates
# # - ballons is used for position updates
# # - all_sprites isused for rendering
# enemies = pygame.sprite.Group()
ballons = pygame.sprite.Group()
body = pygame.sprite.Group()
body.add(left_hand)
body.add(right_hand)
body.add(right_foot)
body.add(left_foot)

score_box = Score_Box()

all_sprites = pygame.sprite.Group()
all_sprites.add(left_hand)
all_sprites.add(right_hand)
all_sprites.add(score_box)
all_sprites.add(right_foot)
all_sprites.add(left_foot)


# all_sprites.add(player)

# Variable to keep our main loop running
running = True
imgsz = 256
stride = 64
dataset = LoadWebcam("0", imgsz, stride )

args, data, model = startup()
args.pose = True
device = select_device(args.device, batch_size=1)

line_color = (255, 0, 0)
score = 0

# Our main loop
counter = 0 
speed = 5
score_box.update_score(score, speed)
while running:
    counter +=1 

    # get an image from the web cam. 
    (_, img, im0, _) = next(iter(dataset))
    print(f"Frame {counter}")
    img = torch.from_numpy(img).to(device)
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    if args.webapp == False:
        out = model(img, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])[0]
        person_dets, kp_dets = run_nms(data, out)
        _, poses, _, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]], person_dets, kp_dets)
    else:
        poses = get_server_response(img)

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

        # Should we add a new cloud?
        elif event.type == ADDBALLON:
            # Create the new cloud, and add it to our sprite groups
            new_ballon = Ballon(speed)
            ballons.add(new_ballon)
            all_sprites.add(new_ballon)

    # # Fill the screen with sky blue
    screen.fill((135, 206, 250))
    
    if args.pose:
        if len(poses)>0:
            pose = poses[0]
            # for pose in poses:
            for seg in data['segments'].values():
                pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
                pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
                # cv2.line(im0, pt1, pt2, args.color_pose, args.line_thick)
                pygame.draw.line(screen, line_color, pt1, pt2, width = 5)
            # pose = poses[0]

            left_wrist, right_wrist = 9, 10
            left_ankle, right_ankle = 15,16
            left_hand.set_center(int(pose[left_wrist,0]), int(pose[left_wrist,1]))
            right_hand.set_center(int(pose[right_wrist,0]), int(pose[right_wrist,1]))

            left_foot.set_center(int(pose[left_ankle,0]), int(pose[left_ankle,1]))
            right_foot.set_center(int(pose[right_ankle,0]), int(pose[right_ankle,1]))




   # print(f"poses shape {poses[0].shape}") 17,3
    # # Check if any enemies have collided with the player
    for body_part in body:
        hit_ballon =  pygame.sprite.spritecollideany(body_part, ballons)
        if hit_ballon:
            hit_ballon.remove = True
            score+=100*body_part.score_multi
            score_box.update_score(score, speed)
            score_box.update()
            collision_sound.play()

      # # Update the position of our enemies and ballons
    # enemies.update()
    ballons.update()

    for b in ballons:
        if b.at_top:
            running = False
            print(f"Final score is {score}")

    # Draw all our sprites
    for entity in all_sprites:
        screen.blit(entity.surf, entity.rect)

 
    if counter % 100 ==99:
        speed += 5
        new_ballon_rate -= 250
        pygame.time.set_timer(ADDBALLON, new_ballon_rate)
        print(f"Making harder. New speed is {speed}")


    # random new ballons on harder levels. 
    if score> 1500:
        ratio = score/5e4
        if np.random.rand()<ratio:
            print(f"Extra ballon ratio = {ratio}")
            new_ballon = Ballon(speed)
            ballons.add(new_ballon)
            all_sprites.add(new_ballon)

 

    # Flip everything to the display
    pygame.display.flip()

    # Ensure we maintain a 30 frames per second rate
    clock.tick(10)

# At this point, we're done, so we can stop and quit the mixer
# pygame.mixer.music.stop()
pygame.mixer.quit()