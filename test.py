import pygame

pygame.init()
pygame.joystick.init()

print("Nombre de manettes :", pygame.joystick.get_count())

for i in range(pygame.joystick.get_count()):
    j = pygame.joystick.Joystick(i)
    j.init()
    print("Nom :", j.get_name())
    print("Axes :", j.get_numaxes())
    print("Boutons :", j.get_numbuttons())
