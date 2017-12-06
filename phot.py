import numpy as np
import ntpath
from copy import deepcopy
from glob import glob
from astropy.stats import mad_std
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from photutils import DAOStarFinder
from photutils import aperture_photometry, CircularAperture, CircularAnnulus
import matplotlib.pyplot as plt
import pytz, datetime
import warnings
warnings.filterwarnings('ignore')

# table[i][0] - star id
# table[i][1] - x position
# table[i][2] - y position
# table[i][3] - flux value

cwd = "/home/connor/Desktop/observing_project/databank/"
n_stars = 3
filter = 'r'


def load_fits(filename):
    hdu_list = fits.open(filename)
    data = hdu_list[0].data
    hdu_list.close()
    return data


def search_names(id1):
    filename_list = []
    for filename in list(set().union(glob(cwd+'*.FIT'), glob(cwd+'*.fit'))):
        if '_'+id1+'_' in filename:
            filename_list.append(filename)
    return filename_list, id1


def onclick(event):
    global fig, ix, iy, coords, cid
    ix, iy = event.xdata, event.ydata
    print 'x = %d, y = %d'%(
        ix, iy)
    coords.append((ix, iy))
    if len(coords) == n_stars:
        fig.canvas.mpl_disconnect(cid)
        plt.close()


def get_mouse_clicks():
    global coords, cid
    coords = []
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(1)
    return coords


def select_stars_closest(coordlist, star_table):
    """Returns information about the stars in the star table closest to the specified (x, y) positions."""
    id_list = []
    for j in range(len(coordlist)):
        x = coordlist[j][0]
        y = coordlist[j][1]
        d_list = []
        for i in range(np.shape(star_table)[0]):
            x2 = np.float64(star_table[i][1])
            y2 = np.float64(star_table[i][2])
            d = np.sqrt((x-x2)**2 + (y-y2)**2)
            d_list.append(d)
        id = d_list.index(np.min(d_list))
        id_list.append(id)
    return star_table[id_list]


def save_geometry(star_table):
    r0x = np.float64(star_table[0][1])
    r0y = np.float64(star_table[0][2])
    print (r0x, r0y)
    coords = [(0.0, 0.0)]
    for i in range(1, np.shape(star_table)[0]):
        rix = np.float64(star_table[i][1])
        riy = np.float64(star_table[i][2])
        print (rix, riy)
        coords.append((rix-r0x, riy-r0y))
    return coords


def match_geometry(shape, star_table):
    dev_list = []

    for i in range(np.shape(star_table)[0]):
        x0, y0 = np.float64(star_table[i][1]), np.float64(star_table[i][2])
        coord_list = [(x0, y0)]
        for j in range(1, np.shape(shape)[0]):
            coord_list.append((x0+shape[j][0], y0+shape[j][1]))
        closest = select_stars_closest(coord_list, star_table)

        distances = []
        for k in range(len(coord_list)):
            x, y = coord_list[k][0], coord_list[k][1]
            x1, y1 = np.float64(closest[k][1]), np.float64(closest[k][2])
            distance = np.sqrt((x1-x)**2 + (y1-y)**2)
            distances.append(distance)

        dev_list.append(np.sum(distances))

    origin_star = star_table[dev_list.index(np.min(dev_list))]
    correct_coords = deepcopy(shape)
    for i in range(np.shape(shape)[0]):
        correct_coords[i] = (correct_coords[i][0] + np.float64(origin_star[1]),
                             correct_coords[i][1] + np.float64(origin_star[2]))

    return select_stars_closest(correct_coords, star_table)


def plot_geometry(data, star_table):
    fig = plt.figure(figsize=(17, 17))
    ax = fig.add_subplot(111)
    ax.imshow(data, cmap='gray', origin='lower', vmin=0, vmax=1500)
    for i in range(0, np.shape(star_table)[0]):
        x = np.float64(star_table[i][1])
        y = np.float64(star_table[i][2])
        ax.scatter(x, y, s=80, facecolors='none', edgecolors='r')
    plt.show()


def star_find(data, fwhm=10., threshold=200.):
    bkg_sigma = mad_std(data)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * bkg_sigma)
    sources = daofind(data)
    return sources


def ap_phot(sources, data, source_r=6., sky_in=15, sky_out=20, plot=False):
    global fig
    centroids = (sources['xcentroid'], sources['ycentroid'])
    source_aperture = CircularAperture(centroids, r=source_r)
    source_area = source_aperture.area()
    source_table = aperture_photometry(data, source_aperture)

    sky_aperture = CircularAnnulus(centroids, r_in=sky_in, r_out=sky_out)
    sky_area = sky_aperture.area()
    sky_table = aperture_photometry(data, sky_aperture)

    sky_subtracted_source = deepcopy(source_table)

    for i in range(np.shape(centroids)[1]):
        sky_value = sky_table[i][3]
        sky_per_pix = sky_value / sky_area
        sky_behind_source = sky_per_pix * source_area
        sky_subtracted_source[i][3] -= sky_behind_source

    if plot:
        fig = plt.figure(figsize=(17, 17))
        plt.imshow(data, cmap='gray', origin='lower', vmin=0, vmax=1500)
        for i in range(np.shape(centroids)[1]):
            plt.annotate(str(source_table[i][0]),
                         xy=(np.float64(source_table[i][1]) + 15.,
                             np.float64(source_table[i][2]) + 15.),
                         color="white")
        source_aperture.plot(color='blue', lw=1.5, alpha=0.5)
        sky_aperture.plot(color="orange", lw=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2)
        plt.close()

    return sky_subtracted_source


def extract_time(filename, filter_id):
    # Extract the date from the filename
    month = ntpath.basename(filename)[2:4]
    day = ntpath.basename(filename)[4:6]
    year = '2017'

    # Extract the time from the time file
    textfile = "a_" + month + day + "_0_" + filter_id + "_time"
    txt = str(np.loadtxt(cwd + textfile, dtype='string'))
    hh = txt[0:2]
    mm = txt[3:5]

    # Convert to UTC
    local = pytz.timezone("US/Eastern")
    naive = datetime.datetime.strptime(year+'-'+month+'-'+day+'T'+hh+':'+mm+':00',
                                       "%Y-%m-%dT%H:%M:%S")
    local_dt = local.localize(naive, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)

    # Build MJD from time and date information
    time_date = utc_dt.strftime('%Y-%m-%dT%H:%M:%S')
    t = Time(time_date, format='isot', scale='utc')
    MJD = t.mjd
    return MJD


# Find all files in directory with specified filter
filesearch = search_names(filter)

filenames = filesearch[0]
filter_id = filesearch[1]

# Remove the first filename from the list, make sure it's set 0
first_filename = ''
for i, file in enumerate(filenames):
    if ntpath.basename(file)[7] == '0':
        first_filename = filenames[i]
del filenames[filenames.index(first_filename)]

# Load in the first fits file
da = fits.open(first_filename)[0].data

# Photometer the first fits file
sources = star_find(da)
photometered = ap_phot(sources, da)
print photometered

# Collect mouse coordinates on first image
fig = plt.figure(figsize=(17, 17))
ax = fig.add_subplot(111)
ax.imshow(da, cmap='gray', origin='lower', vmin=0, vmax=1500)
plt.title("Click "+str(n_stars)+" stars to photometer")
mouse_coords = get_mouse_clicks()

# Set up fits array
column_names = ["filename", "star1_apsum", "star2_apsum", "star3_apsum", "MJD"]
fits_filenames = []
star1_apsums = []
star2_apsums = []
star3_apsums = []
MJDs = []

# Select stars closest to mouse clicks
cut_startable = select_stars_closest(mouse_coords, photometered)

# Obtain MJD for the first file
MJD = extract_time(first_filename, filter_id)

# Add relevant information to fits file arrays
fits_filenames.append(ntpath.basename(first_filename))
star1_apsums.append(cut_startable[0][3])
star2_apsums.append(cut_startable[1][3])
star3_apsums.append(cut_startable[2][3])
MJDs.append(MJD)

# Save the relative geometry for use in the other images
plot_geometry(da, cut_startable)
geometry = save_geometry(cut_startable)

for i, filename in enumerate(filenames):

    # Load in the ith file in the directory
    da = fits.open(filename)[0].data

    # Photometer the ith file
    sources = star_find(da)
    photometered = ap_phot(sources, da)

    # Match the geometry of the first image to the stars in this image, extract stars
    cut_startable = match_geometry(geometry, photometered)

    # Add relevant information to fits file arrays
    fits_filenames.append(ntpath.basename(filename))
    star1_apsums.append(cut_startable[0][3])
    star2_apsums.append(cut_startable[1][3])
    star3_apsums.append(cut_startable[2][3])
    MJDs.append(extract_time(filename, filter_id))

    plot_geometry(da, cut_startable)

t = Table([fits_filenames, star1_apsums, star2_apsums, star3_apsums, MJDs], names=tuple(column_names))
t.write('photometry_'+filter_id+'.fits', format='fits')
