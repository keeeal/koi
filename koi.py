
import sys, time, random, pygame
import numpy as np
from nn import Brain
from ga import GeneticAlgorithm

def blit(screen, sprite, position, angle):
    def rotate_center(image, rect, angle):
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = rot_image.get_rect(center=rect.center)
        return rot_image, rot_rect

    screen.blit(*rotate_center(sprite, sprite.get_rect(center=position), angle))

class Actor(object):
    def __init__(self, position, radius, heading=None):
        super(Actor, self).__init__()
        self.position = list(position)
        self.radius = radius
        self.heading = heading if heading else 360*random.random()

    def __sub__(self, other):
        return self.distance_to(other)

    def distance_to(self, other, ignore_radii=False):
        if isinstance(other, Actor):
            pos, rad = other.position, self.radius + other.radius
        else:
            pos, rad = other, self.rad

        distance = np.linalg.norm(np.subtract(pos, self.position))
        distance = distance if ignore_radii else distance - rad
        return distance

    def angle_to(self, other):
        def unitvec(vector):
            norm = np.linalg.norm(vector)
            if norm == 0:
                return vector
            return vector/norm

        pos = other.position if isinstance(other, Actor) else other
        vec1 = -np.sin(np.radians(self.heading)), -np.cos(np.radians(self.heading))
        vec2 = np.subtract(pos, self.position)
        vec1, vec2 = unitvec(vec1), unitvec(vec2)
        angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
        left = np.cross(vec1, vec2) > 0
        return -np.degrees(angle) if left else np.degrees(angle)

class Fish(Actor):
    speed, ttl, max_ttl = 2, 2000, 20000
    joints = [(8,37),(14,24),(26,26),(15,15),(17,3),(0,0)]
    dgaus_x = np.arange(-np.pi, np.pi, 0.05*np.pi)
    dgaus = 1.2*dgaus_x*pow(np.e, 0.5)*np.exp(-np.power(dgaus_x, 2)/2)
    input_size = 21

    def __init__(self, position, brain=None):
        super(Fish, self).__init__(position, 34)
        self.segs = 6*[self.heading]
        self.last_turn = 1
        self.dgaus = []
        self.food_eaten = 0
        self.ttl = Fish.ttl
        self.max_ttl = Fish.max_ttl

        if brain:
            self.brain = brain
        else:
            self.brain = Brain(Fish.input_size)

    def tick(self, pond):
        if self.ttl < 0 or self.max_ttl < 0:
            pond.dead += 1

            if self.ttl < 0:
                print('A fish died of starvation. Total dead:', pond.dead)

            if self.max_ttl < 0:
                print('A fish died of old age. Total dead:', pond.dead)

            pond.fish.remove(self)
            self.brain.fitness.values = self.food_eaten,
            return self.brain

        input_vector = Fish.input_size*[0]

        for food in pond.food:
            distance = self.distance_to(food, ignore_radii=True)

            if distance - self.radius < food.ripple.radius:
                angle = self.angle_to(food)

                if abs(angle) < 360*(1/32):
                    if distance < 100:
                        input_vector[0] = 1
                    elif distance < 300:
                        input_vector[1] = 1
                    else:
                        input_vector[2] = 1
                elif abs(angle) < 360*(1/8):
                    if angle < 0:
                        if distance < 100:
                            input_vector[3] = 1
                        elif distance < 300:
                            input_vector[4] = 1
                        else:
                            input_vector[5] = 1
                    else:
                        if distance < 100:
                            input_vector[6] = 1
                        elif distance < 300:
                            input_vector[7] = 1
                        else:
                            input_vector[8] = 1
                elif abs(angle) < 360*(1/4):
                    if angle < 0:
                        if distance < 100:
                            input_vector[9] = 1
                        elif distance < 300:
                            input_vector[10] = 1
                        else:
                            input_vector[11] = 1
                    else:
                        if distance < 100:
                            input_vector[12] = 1
                        elif distance < 300:
                            input_vector[13] = 1
                        else:
                            input_vector[14] = 1
                else:
                    if angle < 0:
                        if distance < 100:
                            input_vector[15] = 1
                        elif distance < 300:
                            input_vector[16] = 1
                        else:
                            input_vector[17] = 1
                    else:
                        if distance < 100:
                            input_vector[18] = 1
                        elif distance < 300:
                            input_vector[19] = 1
                        else:
                            input_vector[20] = 1

        self.move(*self.brain(input_vector))

        if len(pond.food):
            closest_food = min(pond.food, key=self.distance_to)

            if self.distance_to(closest_food) < 0:
                self.food_eaten += 1
                self.ttl += 200
                pond.food.remove(closest_food)

        self.ttl -= 1
        self.max_ttl -= 1

    def move(self, turn, forward):
        turn = min(max(turn, -1), 1)
        forward = min(max(forward, 0), 1)
        self.heading += turn
        self.last_turn = -1 if turn < 0 else 1
        self.position[0] -= forward*Fish.speed*np.sin(np.radians(self.heading))
        self.position[1] -= forward*Fish.speed*np.cos(np.radians(self.heading))

    def draw(self, screen, simple=False):
        if simple:
            triangle = [(self.position[0] - (2/3)*self.radius*np.sin(np.radians(self.heading)),
                         self.position[1] - (2/3)*self.radius*np.cos(np.radians(self.heading))),
                        (self.position[0] - (1/3)*self.radius*np.sin(np.radians(self.heading + 30)),
                         self.position[1] - (1/3)*self.radius*np.cos(np.radians(self.heading + 30))),
                        (self.position[0] - (1/3)*self.radius*np.sin(np.radians(self.heading - 30)),
                         self.position[1] - (1/3)*self.radius*np.cos(np.radians(self.heading - 30)))]

            pygame.draw.polygon(screen, (139, 0, 0), triangle, 2)
            pygame.draw.circle(screen, (139, 0, 0),
                    (int(self.position[0]), int(self.position[1])),
                     int(self.radius), 2)
        else:
            pos = joint = self.position
            self.segs[0] = self.heading

            if len(self.dgaus) < 5*len(self.segs):
                self.dgaus += int(300*random.random())*[0]
                self.dgaus += list(-self.last_turn*Fish.dgaus)

            for i in range(len(self.segs)):
                self.segs[len(self.segs)-i-1] = \
                    0.5*pow(len(self.segs)-i-1, 1.5)*self.dgaus[5*i] + self.segs[len(self.segs)-i-1]

            self.dgaus = self.dgaus[1:]

            for k in range(len(self.segs)):
                if k:
                    self.segs[k] = ((4*k + 15)*self.segs[k] + self.segs[k - 1])/(4*k + 16)
                    pos = joint[0] + Fish.joints[k - 1][1]*np.sin(np.radians(self.segs[k])), \
                          joint[1] + Fish.joints[k - 1][1]*np.cos(np.radians(self.segs[k]))

                joint = pos[0] + Fish.joints[k][0]*np.sin(np.radians(self.segs[k])), \
                        pos[1] + Fish.joints[k][0]*np.cos(np.radians(self.segs[k]))

                blit(screen, Fish.sprite[k], pos, self.segs[k])

class Food(Actor):
    def __init__(self, position):
        super(Food, self).__init__(position, 24)
        self.ripple = Ripple(self.position)

    def draw(self, screen, simple=False):
        if simple:
            pygame.draw.circle(screen, (245, 208, 76),
                    [int(p) for p in self.position], int(self.radius), 2)
        else:
            blit(screen, Food.sprite, self.position, self.heading)

class Ripple(Actor):
    def __init__(self, position):
        super(Ripple, self).__init__(position, 24)

    def tick(self, pond):
        if self.radius > max(pond.size):
            pond.ripples.remove(self)

        self.radius += 2

    def draw(self, screen, simple=False):
        pygame.draw.circle(screen, (255, 255, 255),
                [int(p) for p in self.position], int(self.radius), 2)

class Pond(object):
    def __init__(self, size, n_food):
        super(Pond, self).__init__()
        self.screen = None
        self.size = self.width, self.height = size
        self.n_food = n_food
        self.fish, self.food, self.ripples = [], [], []
        self.dead = 0

    def add_fish(self, brain=None):
        x, y = self.width*random.random(), self.height*random.random()
        self.fish.append(Fish((x, y), brain=brain))

    def add_food(self):
        x, y = self.width*random.random(), self.height*random.random()
        self.food.append(Food((x, y)))
        self.ripples.append(self.food[-1].ripple)

    def tick(self):
        while len(self.food) < self.n_food:
            self.add_food()

        dead_fish = []

        for ripple in self.ripples:
            ripple.tick(self)

        for fish in self.fish:
            dead = fish.tick(self)

            if dead:
                dead_fish.append(dead)

        return dead_fish

    def init_screen(self, simple=False):
        pygame.init()

        if not simple:
            self.background = pygame.image.load('img/bg.jpg')
            self.foreground = pygame.image.load('img/fg.png')
            Fish.sprite = [pygame.image.load('img/koi' + str(i) + '.png') for i in range(6)]
            Food.sprite = pygame.image.load('img/food.png')

        self.screen = pygame.display.set_mode(self.size)

    def draw(self, simple=False):
        if not self.screen:
            self.init_screen()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        if simple:
            self.screen.fill((133, 183, 193))
        else:
            self.screen.blit(self.background, (0,0))

        for actor in self.fish + self.food + self.ripples:
            actor.draw(self.screen, simple=simple)

        pygame.display.update()

def main(n_fish=8, n_food=8, n_pop=64, draw=2):
    if draw == 1:
        draw, simple = True, True
    elif draw == 2:
        draw, simple = True, False

    GA = GeneticAlgorithm(Fish.input_size, n_pop=n_pop)
    pond = Pond((1280, 720), n_food=n_food)

    while True:
        while len(pond.fish) < n_fish:
            pond.add_fish(GA.next())

        dead = pond.tick()

        if len(dead):
            GA.graveyard.update(dead)

        if draw:
            pond.draw(simple=simple)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fish', dest='n_fish', type=int, default=8,
        help='Number of fish on the screen at once.')
    parser.add_argument('--food', dest='n_food', type=int, default=8,
        help='Number of food on the screen at once.')
    parser.add_argument('--pop', dest='n_pop', type=int, default=64,
        help='Genetic algorithm queue and graveyard size.')
    parser.add_argument('--draw', type=int, default=2, choices=(0,1,2),
        help='Draw mode. 0: Do not draw, 1: Simple shapes, 2: Full graphics.')
    main(**vars(parser.parse_args()))
