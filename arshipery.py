#!/usr/bin/env python3
# encoding: utf-8

# Playing battleships with archery.
# You fire two arrows, one for the column index and one for the line index.
# The score on the target gives you the index.
# An arrow going out of the target discards the current pair.

# Hypothesis:
# – The arrow distribution on the target follow a normal law.
# – The drift on target is symetric in all direction. There is no covariance in the distribution.

# Statistics questions:
# – Given a player's precision, what are the cells with the minimal probability of hit?
#       A: 10/10, you'd better place ships on the 10th line and column.
# – What if players have different levels?

# Game theory questions:
# – What if players have a limited number of arrows?
# – What are the optimal mixed strategy (for ship placement and aiming)?


import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

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


def plot_targets(ax, targets):
    nb_circles = len(targets)-1

    # Pastel target colors
    # targets_colors = ["gold","lightcoral","lightblue","lightgrey","white",]

    # Official targets color
    targets_colors = ["yellow","red","blue","black","white",]

    prev_r = 0
    for t,(inf,sup) in enumerate(targets):
        face   =  plt.Circle((0, 0), sup, color=targets_colors[(nb_circles-t-1)//2], zorder=1, linewidth=0.2)
        # border =  plt.Circle((0, 0), sup, color="grey", fill=False,zorder=2)
        ax.add_artist(face)
        # ax.add_artist(border)
        ax.set_aspect("equal")
    # last circle
    last_border =  plt.Circle((0, 0), targets[0][1], color="grey", fill=False, zorder=2)
    ax.add_artist(last_border)


def format_grid(im, ax, nb_circles):
    # Grid
    minticks = [i-0.5 for i in range(nb_circles+1)]
    ax.set_xticks(minticks, minor=True)
    ax.set_yticks(minticks, minor=True)
    ax.grid(which="minor")

    # Labels
    majticks = [i for i in range(nb_circles)]
    labels = [i+1 for i in range(nb_circles)]
    ax.set_xticks(majticks)
    ax.set_yticks(majticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for tick in ax.xaxis.get_major_ticks():
        # tick.label.set_fontsize(8)
        tick.label.set_fontsize(5)

    for tick in ax.yaxis.get_major_ticks():
        # tick.label.set_fontsize(8)
        tick.label.set_fontsize(5)

    # FIXME add color bars
    aspect = 20
    pad_fraction = 0.5
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1./aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5)
    # v = np.linspace(0, 0.03, 3, endpoint=True)
    # cbar.ax.set_yticklabels(["{:1.2f}".format(i) for i in v])

    # fig1.colorbar(im, ax=axarr[1,i])


if __name__ == "__main__":

    if len(sys.argv) > 1:
        nb_arrows = int(sys.argv[1])
    else:
        nb_arrows = 10000

    d_radius = 10
    dim = 2
    nb_circles = 10
    player_level = {"01-gold":1, "02-yellow":3,"04-red":5,"06-blue":9,"08-black":13,"10-white":17,"100-crap":25,"900-wtf":40}
    nb_lvl = len(player_level)
    print("levels=",sorted(player_level.keys()))

    targets = make_target(nb_circles, d_radius)
    print("targets=",targets)

    # One column per player level, two subplots:
    # oup, an example target, down the density of probability.
    fig1, axarr = plt.subplots(nb_lvl+1,nb_lvl+1)

    lvl_score = {}
    k = 0
    for i,pl1 in enumerate(sorted(player_level.keys())):
        for j,pl2 in enumerate(sorted(player_level.keys())):
            fi, fj = i+1, j+1
            print(k,"/",nb_lvl*nb_lvl,":",pl1,"VS",pl2)
            k+=1
            sys.stdout.flush()

            # FIRE ARROWS
            arrows, score = play(player_level[pl1], player_level[pl2], nb_circles, nb_arrows, dim)
            lvl_score[(pl1,pl2)] = score

            # PLOT TARGET
            plot_targets(axarr[0,fj], targets)
            plot_targets(axarr[fi,0], targets)

            # PLOT (some) ARROWS
            if i==j:
                aim_p1, aim_p2 = 10,10
                max_points = 100
                lim = 1.5 * (nb_circles * d_radius)
                axarr[0,fj].set_xlim((-lim,lim))
                axarr[fi,0].set_xlim((-lim,lim))
                axarr[0,fj].set_ylim((-lim,lim))
                axarr[fi,0].set_ylim((-lim,lim))
                p1,p2 = 0,1
                p1_arrows = arrows[aim_p1-1][aim_p2-1][p1][:,:max_points]
                p2_arrows = arrows[aim_p1-1][aim_p2-1][p2][:,:max_points]

                axarr[0,fj].scatter(*    p1_arrows , edgecolor="green", color="green", alpha=0.9, marker=".", zorder=3)
                axarr[fi,0].scatter(*(-1*p2_arrows), edgecolor="green", color="green", alpha=0.9, marker=".", zorder=3)

                # axarr[fi,0].set_title(pl1.split("-")[1])
                axarr[fi,0].text(-200, 0, pl1.split("-")[1], ha='center', va='center', rotation='vertical')
                axarr[0,fj].set_title(pl2.split("-")[1])

                axarr[0,fj].axes.get_yaxis().set_visible(False)
                axarr[fi,0].axes.get_yaxis().set_visible(False)
                axarr[0,fj].axes.get_xaxis().set_visible(False)
                axarr[fi,0].axes.get_xaxis().set_visible(False)


            # PLOT proba maps

            # Compute probabilities of hit
            H,xe,ye = np.histogram2d(*score, bins=11, normed=True)

            # Plot the normalized histogram without out arrows.
            if i==j:
                colormap = "cubehelix"
            else:
                colormap = "viridis"

            im = axarr[fi,fj].imshow(H[1:,1:], interpolation='nearest', origin='low', cmap=colormap,)
            format_grid(im, axarr[fi,fj], nb_circles)


    # PLOT RANDOM TEAMS
    # Draw random player levels couples
    nb_games = int(math.sqrt(nb_arrows)*10)
    nb_subarrows = max(int(math.sqrt(nb_arrows)*10), nb_arrows)
    pl1 = [int(i) for i in np.random.randn(nb_games) + len(player_level)//2]
    pl2 = [int(i) for i in np.random.randn(nb_games) + len(player_level)//2]
    for i in range(nb_games):
        if 0 > pl1[i] > len(player_level):
            pl1[i] = np.random.uniform(0,len(player_level))
        if 0 > pl2[i] > len(player_level):
            pl2[i] = np.random.uniform(0,len(player_level))

    p1,p2 = 0,1
    score1 = []
    score2 = []
    for i in range(nb_games):
        print(i,"/",nb_games,end="\r")
        k1, k2 = (pl1[i], pl2[i])
        lvl_pair = list(player_level.keys())[k1], list(player_level.keys())[k2]
        full_score = lvl_score[lvl_pair]
        score1 = np.concatenate( (score1, np.random.choice(full_score[p1], (nb_subarrows))) )
        score2 = np.concatenate( (score2, np.random.choice(full_score[p2], (nb_subarrows))) )

    H,xe,ye = np.histogram2d(score1, score2, bins=11, normed=True)
    im = axarr[0,0].imshow(H[1:,1:], interpolation='nearest', origin='low', cmap="inferno",)
    format_grid(im, axarr[0,0], nb_circles)


    # s = 1000
    # h,w = s * 3, s * 4
    # plt.savefig("target_eq-levels.png",dpi=300, figsize=(h,w))

    plt.show()
