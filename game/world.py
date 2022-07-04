# A world has ballons going up

# from turtle import speed
import cv


class World:
    def __init__(self, screen_size) -> None:
        self.ballons = []
        self.screen_size = screen_size

class Moving_Object:
    speed = 10

    def step(self):
        self.y-=self.speed

    def draw(self,image_input):
        pass


class Ballon(Moving_Object):
    def __init__(self, x,y, speed) -> None:
        self.x
        self.y
        self.speed = speed
        self.sprite = cv.imread('sprite/ballon.jpg')
        # self.sprite.re        
        self.shape = (100,100) # x,y
        self.half = (100/2,100/2)

        self.sprite = cv.resize(self.sprite,self.shape)

    def draw(self, image_input):
        input_size = image_input.shape

        # xmin, ymin, xmax, ymax
        target_position = (self.x - self.half[0],
                            self.y- self.half[1],
                            self.x + self.half[0], 
                            self.y+ self.half[1] )

        actual_position = (max(0, target_position[0]), 
                            max(0,target_position[1]), 
                            min(input_size[1], target_position[2]), 
                            min(input_size[0]), target_position[3])

        # if actual_position[]
        image_input[actual_position[1]: actual_position[3], 
                    actual_position[0]: actual_position[2]] = self.sprite                             


