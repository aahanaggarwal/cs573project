#%%

from numpy.lib.type_check import imag
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

locations = []
x, y = [], []

data = json.load(open('shots.json', 'r'))

for shot in data:
    locations.append(shot['location'])
    x.append(shot['location'][0])
    y.append(shot['location'][1])

locations = np.array(locations)
x = np.array(x)
y = np.array(y)
print(locations, x, y)


#%%

# remove axis
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

heatmap, xedges, yedges = np.histogram2d(x, y, bins=70)

# remove some unnecessary blank space
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.ylim(extent[2]+30, extent[3])



plt.imshow(heatmap, extent=extent, origin='lower', cmap='hot')

# plt.savefig('media/heatmap_of_shot_locations.png')
plt.show()


# %%

key_pass_data = json.load(open('key_passes.json', 'r'))
key_pass_array = []
for key_pass in key_pass_data:
    key_pass_array.append(key_pass['id'])

shot_x = []
shot_y = []
pass_x = []
pass_y = []
for shot in data:
    if 'key_pass_id' in shot['shot'] and shot['shot']['key_pass_id'] in key_pass_array:
        shot_x.append(shot['location'][0])
        shot_y.append(shot['location'][1])
        key_pass = key_pass_data[key_pass_array.index(shot['shot']['key_pass_id'])]
        pass_x.append(key_pass['location'][0])
        pass_y.append(key_pass['location'][1])
# %%
print(shot_x[0], shot_y[0], pass_x[0], pass_y[0])

shot_x = np.array(shot_x)
shot_y = np.array(shot_y)
pass_x = np.array(pass_x)
pass_y = np.array(pass_y)

plt.clf()

# this doesnt look great
plt.plot([pass_x, shot_x], [pass_y, shot_y])


#%%

import datashader as ds, colorcet

# add some noise to x and y
x_noisy = x + np.random.normal(0, 0.25, len(x))
y_noisy = y + np.random.normal(0, 0.25, len(y))


x_clipped = x_noisy.clip(60, 120)

df = pd.DataFrame({'x': x_clipped, 'y': y_noisy})

cvs = ds.Canvas(plot_width=400, plot_height=300)
agg = cvs.points(df, 'x', 'y')
img = ds.tf.shade(agg, how='cbrt', cmap=colorcet.fire)
img = ds.tf.set_background(img, 'black')
pil_img = img.to_pil()
pil_img.save('media/heatmap_of_shot_locations_ds.png')
pil_img.show()
# Dont really wanna save this

#%%

df = pd.DataFrame({'pass_x': pass_x, 'pass_y': pass_y, 'shot_x': shot_x, 'shot_y': shot_y})
cvs = ds.Canvas(plot_width=400, plot_height=300)
agg = cvs.line(df, x='pass_x', y='pass_y')
img = ds.tf.shade(agg, how='cbrt', cmap=colorcet.fire)
img = ds.tf.set_background(img, 'black')


# agg = cvs.line(df, x=['A1', 'A2'], y=['B1', 'B2'], axis=0) # doctest: +SKIP
# ... tf.spread(tf.shade(agg))
# %%

def getEquidistantPoints(p1, p2, parts):
    return list(zip(np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1)))

all_points = []
print(len(pass_x))
for i in range(0, len(pass_x)):
    all_points.extend(list(getEquidistantPoints((pass_x[i], pass_y[i]), (shot_x[i], shot_y[i]), 500)))

print(len(all_points))
all_points = np.array(all_points)
print(all_points)

# %%

all_x = all_points[:, 0]
all_y = all_points[:, 1]
print(all_x)

df = pd.DataFrame({'x': all_x, 'y': all_y})

cvs = ds.Canvas(plot_width=400, plot_height=300)
agg = cvs.points(df, 'x', 'y')
img = ds.tf.shade(agg, how='log', cmap=colorcet.fire)
img = ds.tf.set_background(img, 'black')
pil_img = img.to_pil()
pil_img.save('media/heatmap_of_key_passes.png')
pil_img.show()
# %%
