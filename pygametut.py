import pygame as pg
import numpy as np
import neat
import os

SCR_WIDTH = 1200
SCR_HEIGHT = 600




class bird:
	def __init__(self,x,y,img,s):
		self.x = x
		self.y = y
		self.img = pg.image.load(img)
		self.img = pg.transform.scale(self.img,(60,60))
		self.vel = 0
		self.hitbox = pg.Rect(x+15,y+20,30,20)
		self.scr = s
		self.score = 0
		self.allow = True
	def show(self):
		self.scr.blit(self.img,(self.x,self.y))
	def apply_gravity(self):
		self.vel += 0.009
	def update(self):
		self.y += self.vel 
		self.hitbox.y = self.y+20
	def jump(self):
		self.vel = -0.99

class pipe:
	GAP = 150
	VEL = -0.3
	def __init__(self,x,h,s):
		self.x = x
		self.height = h
		self.image = pg.image.load('pillar.png')
		self.hitbox1 = pg.Rect(x,SCR_HEIGHT-h,50,h)
		self.hitbox2 = pg.Rect(x,0,50,SCR_HEIGHT-(self.height+self.GAP))
		self.scr = s
	def show(self):
		img = pg.transform.scale(self.image,(50,self.height))
		self.scr.blit(img,(self.x,SCR_HEIGHT-self.height))
		img = pg.transform.flip(self.image,False,True)
		height = SCR_HEIGHT-(self.height+self.GAP)
		img = pg.transform.scale(img,(50,height))
		self.scr.blit(img,(self.x,0))
	def update(self):
		self.x += self.VEL
		self.hitbox1.x = self.x
		self.hitbox2.x = self.x




def check_pillar(p,s,players):
	a = np.random.randint(80,400,size = 1)
	if p[0].x<=-50:
		p[0] = p[1]
		p[1] = pipe(SCR_WIDTH,a[0],s) 
		for i in players:
			if i.allow:
				i.score += 50

def check_collision(p,player):
	if player.hitbox.colliderect(p[0].hitbox1) or player.hitbox.colliderect(p[0].hitbox2):
		return True
	else:
		if player.hitbox.y<=0 or player.hitbox.y>= (SCR_HEIGHT-50)-player.hitbox.h:
			return True
		else:
			return False




def main(genomes,config):
	pg.init()
	for i in range(0,len(genomes)):
		genomes[i][1].fitness = 0
	scr = pg.display.set_mode((SCR_WIDTH,SCR_HEIGHT))

	p = [pipe(600,np.random.randint(80,400,size = 1)[0],scr),pipe(1200,np.random.randint(80,400,size = 1)[0],scr)]
	players = [bird(50,50,'icon.png',scr) for _ in range(0,100)]
	

	running = True

	
	ground = pg.transform.scale(pg.image.load('bottom.png'),(SCR_WIDTH,50))
	left = 100
	while running:
		for event in pg.event.get():
			if event.type == pg.QUIT:
				running = False
			if event.type == pg.KEYDOWN:
				if event.key == pg.K_SPACE:
					player.jump()
		scr.fill((0,0,0))
		
		for i in p:
			i.show()
			i.update()
		check_pillar(p,scr,players)
		scr.blit(ground,(0,SCR_HEIGHT-50))
		for player,(idg,genome) in zip(players,genomes):
			if not player.allow:
				continue
			net = neat.nn.FeedForwardNetwork.create(genome, config)
			pg.draw.rect(scr,(255,0,0),p[0].hitbox2,3)
			if net.activate((SCR_HEIGHT-player.y,p[0].height,p[0].height+pipe.GAP))[0]>0.5:
				player.jump()
			player.show()
			player.update()
			player.apply_gravity()
			if check_collision(p,player):
				player.allow = False
				genome.fitness = player.score
				left -= 1
			player.score += 0.01
		pg.display.update()
		if left == 0:
			break
	pg.quit()


def run(config_file):
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
	P = neat.Population(config)
	P.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	P.add_reporter(stats)
	winner = P.run(main,50)

if __name__ == "__main__":
	local_dir = os.path.dirname(__file__)
	config_file = os.path.join(local_dir,'configure.txt')
	run(config_file)
