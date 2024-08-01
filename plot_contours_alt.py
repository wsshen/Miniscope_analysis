def plot_contours(A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
              cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None, coordinates=None,
              contour_args={}, number_args={}, **kwargs):
"""Plots contour of spatial components against a background image and returns their coordinates

 Args:
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)

     Cn:  np.ndarray (2D)
               Background image (e.g. mean, correlation)

     thr_method: [optional] string
              Method of thresholding:
                  'max' sets to zero pixels that have value less than a fraction of the max value
                  'nrg' keeps the pixels that contribute up to a specified fraction of the energy

     maxthr: [optional] scalar
                Threshold of max value

     nrgthr: [optional] scalar
                Threshold of energy

     thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)
               Kept for backwards compatibility. If not None then thr_method = 'nrg', and nrgthr = thr

     display_number:     Boolean
               Display number of ROIs if checked (default True)

     max_number:    int
               Display the number for only the first max_number components (default None, display all numbers)

     cmap:     string
               User specifies the colormap (default None, default colormap)

 Returns:
      coordinates: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
"""

if swap_dim:
    Cn = Cn.T
    print('Swapping dim')

if thr is None:
    try:
        thr = {'nrg': nrgthr, 'max': maxthr}[thr_method]
    except KeyError:
        thr = maxthr
else:
    thr_method = 'nrg'

for key in ['c', 'colors', 'line_color']:
    if key in kwargs.keys():
        color = kwargs[key]
        kwargs.pop(key)

ax = pl.gca()
if vmax is None and vmin is None:
    pl.imshow(Cn, interpolation=None, cmap=cmap,
              vmin=np.percentile(Cn[~np.isnan(Cn)], 1),
              vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
else:
    pl.imshow(Cn, interpolation=None, cmap=cmap, vmin=vmin, vmax=vmax)

if coordinates is None:
    coordinates = get_contours(A, np.shape(Cn), thr, thr_method, swap_dim)
for c in coordinates:
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    pl.plot(*v.T, c=colors, **contour_args)

if display_numbers:
    d1, d2 = np.shape(Cn)
    d, nr = np.shape(A)
    cm = com(A, d1, d2)
    if max_number is None:
        max_number = A.shape[1]
    for i in range(np.minimum(nr, max_number)):
        if swap_dim:
            ax.text(cm[i, 0], cm[i, 1], str(i + 1), color=colors, **number_args)
        else:
            ax.text(cm[i, 1], cm[i, 0], str(i + 1), color=colors, **number_args)
return coordinates