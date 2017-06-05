
# Playing battleships with archery.
# You fire two arrows, one for the column index and one for the line index.
# The score on the target gives you the index.
# An arrow going out of the target discards the current pair.

# Hypothesis:
# – The arrow distribution on the target follow a normal law.
# – The drift on target is symetric in all direction. There is no covariance in the distribution.

# Questions:
# – Given a player's precision, what are the cells with the minimal probability of hit?

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

def x(p):
    return p[0]
def y(p):
    return p[1]
def dist(u,v):
    return np.sqrt( (x(v)-x(u))*(x(v)-x(u)) + (y(v)-y(u))*(y(v)-y(u)) )


def fire(nb_arrows, targeted_score, dispersion_score, dim=2):
    """Return a numpy array of arrows coordinates the form: [[x_1, … , x_n],[y_1, …, y_n]]"""
    assert( 1 <= targeted_score <= 10)
    # Variance of 1 = almost all arrows in center.
    mean = np.zeros(nb_arrows) + (10 - targeted_score) * d_radius / math.sqrt(2)
    var  = np.ones(nb_arrows) * 2*dispersion_score # FIXME var = f(disp,d_radius)
    arrows=np.random.normal(mean, var, (dim,nb_arrows))
    return arrows


def make_target(nb_circles, d_radius):
    """Return an array of circles radius ranges of he form: [(inf_10,sup_10), …, (inf_1,sup_1)]"""
    target_dist = []
    prev_r = float("inf")
    for t in np.arange( nb_circles, 0, -1 ):
        r = t*d_radius-d_radius/2
        target_dist.append((prev_r,r))
        prev_r = r
    target_dist.append((r,0))
    return target_dist


def play(p1_level, p2_level, nb_circles, nb_arrows, dim = 2):
    center = np.zeros((dim,nb_arrows))
    p1,p2 = 0,1
    score = np.zeros((2,nb_arrows*nb_circles*nb_circles))
    arrows = []
    for i in range(nb_circles):
        arrows.append([None for i in range(nb_circles)])
    # The players target (i,j)
    n = 0
    for i in range(nb_circles):
        for j in range(nb_circles):
            arrows[i][j] = [ fire(nb_arrows, i+1, p1_level, dim), fire(nb_arrows, j+1, p2_level, dim) ]
            dists = [ dist(arrows[i][j][p1],center), dist(arrows[i][j][p2],center) ]

            # Compute the scores reached by each arrow.
            for a in range(nb_arrows):
                d1 = dists[p1][a]
                d2 = dists[p2][a]
                # If at one of the arrows is out, let a score of zero for both. # FIXME option to discard zeros
                if d1 < targets[0][1] and d2 < targets[0][1]:
                    # Find the reached circle and mark score.
                    for k,(sup,inf) in enumerate(targets):
                        if inf <= d1 < sup:
                            score[p1][n] = k
                        if inf <= d2 < sup:
                            score[p2][n] = k
                n += 1

    return arrows, score


if __name__ == "__main__":

    nb_arrows = 10000

    d_radius = 10
    dim = 2
    nb_circles = 10
    player_level = {"01-gold":1, "02-yellow":2,"04-red":5,"06-blue":9,"08-black":11,"10-white":14,"99-crap":20}
    print("levels=",sorted(player_level.keys()))

    targets = make_target(nb_circles, d_radius)
    print("targets=",targets)

    # One line per player level, two subplots:
    # on left, an example target, on right the density of probability.
    fig, axarr = plt.subplots(len(player_level),3)


    for i,pl in enumerate(sorted(player_level.keys())):
        print(i,"/",len(player_level),":",pl)
        sys.stdout.flush()

        # DRAW TARGET

        # Pastel
        targets_colors = ["gold","lightcoral","lightblue","lightgrey","white",]

        # Official
        # targets_colors = ["yellow","red","blue","black","white",]

        prev_r = 0
        for t,(inf,sup) in enumerate(targets):
            face   =  plt.Circle((0, 0), sup, color=targets_colors[(nb_circles-t-1)//2],zorder=2)
            border =  plt.Circle((0, 0), sup, color="grey", fill=False,zorder=2)
            axarr[i,0].add_artist(face)
            axarr[i,0].add_artist(border)
            axarr[i,0].set_aspect("equal")


        # DRAW ARROWS
        # for i,k in enumerate(player_level.keys()):
        #     arrows = fire(nb_arrows, 10-i, player_level[k])
        #     axarr[0,0].scatter(*arrows, edgecolor=player_color[i], color="none", alpha=0.3, marker=".", zorder=100)

        arrows, score = play(player_level[pl], player_level[pl], nb_circles, nb_arrows, dim)


        # Plot arbitrary arrows.
        aim_p1 = 1
        aim_p2 = 1
        max_points = 100
        lim = 2 * (nb_circles * d_radius)
        axarr[i,0].set_xlim((-lim,lim))
        axarr[i,0].set_ylim((-lim,lim))
        p1,p2 = 0,1
        p1_arrows = arrows[aim_p1-1][aim_p2-1][p1][:,:max_points]
        p2_arrows = arrows[aim_p1-1][aim_p2-1][p2][:,:max_points]
        axarr[i,0].scatter(*    p1_arrows , edgecolor="magenta", color="none", alpha=0.3, marker=".", zorder=100)
        axarr[i,0].scatter(*(-1*p2_arrows), edgecolor="green"  , color="none", alpha=0.3, marker=".", zorder=100)

        H,xe,ye = np.histogram2d(*score, bins=11, normed=True)

        # Plot the full normalized histogram
        axarr[i,1].imshow(H, interpolation='nearest', origin='low',
                # extent=[xe[0],xe[-1],ye[0],ye[-1]]
                # extent=[0,11,0,11]
        )

        # Plot the normalized histogram without out arrows.
        axarr[i,2].imshow(H[1:,1:], interpolation='nearest', origin='low',
                # extent=[xe[0],xe[-1],ye[0],ye[-1]]
                # extent=[1,11,1,11]
        )

    plt.show()

