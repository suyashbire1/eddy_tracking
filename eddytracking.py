import numpy as np
from netCDF4 import Dataset as dset, MFDataset as mfdset
from scipy.spatial import Delaunay
import time
import pickle
from toolz import groupby
import multiprocessing
#import pymom6.pymom6 as pym6
from matplotlib._contour import QuadContourGenerator
import matplotlib as mpl
#mv = pym6.MOM6Variable


def _distance(x1, y1, x2, y2):
    """Calculates the distance between two points"""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


class Eddy:
    """A class to hold information about detected eddies"""

    #    def __init__(self,identifier,ctr,x,y,xkm,ykm,r,rzeta,rzetac,e,ec,ssha,sshac,t):
    def __init__(self, *args, **kwargs):
        """Initializes an eddy track"""
        tentative_eddy = kwargs.get('tentative_eddy', None)
        if tentative_eddy is None:
            for k, v in kwargs.items:
                setattr(self, k, v)
        else:
            self.init_from_tentative(tentative_eddy)
        self._id_ = None
        self._track_id = None

    def __repr__(self):
        return 'Eddy {} at time {}'.format(self._id_, self.time)

    def init_from_tentative(self, tentative_eddy):
        self.ctr = tentative_eddy.ctr
        self.time = tentative_eddy.time
        self.r = tentative_eddy.r
        self.x = tentative_eddy.x
        self.y = tentative_eddy.y
        self.xkm = tentative_eddy.xkm
        self.ykm = tentative_eddy.ykm
        self.xc = tentative_eddy.xc
        self.yc = tentative_eddy.yc
        self.xckm = tentative_eddy.xckm
        self.yckm = tentative_eddy.yckm
        self.rzeta = tentative_eddy.rzeta
        self.ssha = tentative_eddy.ssha
        self.variables = tentative_eddy.variables
        self.mean_variables = tentative_eddy.mean_variables

    @property
    def id_(self):
        return self._id_

    @id_.setter
    def id_(self, id_):
        self._id_ = id_

    @property
    def track_id(self):
        return self._track_id

    @track_id.setter
    def track_id(self, track_id):
        self._track_id = track_id


class TentativeEddy:
    """A class to hold a contour and methods to check if the
    contour contains eddies."""

    def __init__(self, ctr, time, variables, domain, mean_variables):
        self.domain = domain
        self.time = time
        self.variables = variables
        self.mean_variables = mean_variables
        self.ctr = ctr
        self.is_eddy = self.is_contour_open()
        if self.is_eddy:
            self.is_eddy = self.is_radius_too_extreme()
        if self.is_eddy:
            self.is_eddy = self.is_contour_not_circular()
        if self.is_eddy:
            self.is_eddy = self.is_vort_sign_homogeneous()
        if self.is_eddy:
            self.get_max_ssha()

    def is_contour_open(self):
        """Returns true if contour is closed"""
        ctr = self.ctr
        dtype = ctr.dtype.descr * 2
        return not (ctr.view(dtype).shape[0] == np.unique(
            ctr.view(dtype)).shape[0])

    @staticmethod
    def convert_contour_to_km(ctr):
        """Converts vertices from degree to km"""
        R = 6378
        x, y = ctr[:, 0], ctr[:, 1]
        ykm = R * np.radians(y)
        xkm = R * np.cos(np.radians(y)) * np.radians(x)
        return x, y, xkm, ykm

    def is_radius_too_extreme(self, rmin=15, rmax=150):
        """Retruns the effective radius, a boolean
        which is true if radius of contour is within
        reasonable range (default 15 to 150 km), and
        coordinates of the contour in km"""
        x, y, xkm, ykm = self.convert_contour_to_km(self.ctr)
        area = 0.5 * np.fabs(np.dot(xkm, np.roll(ykm, -1) - np.roll(ykm, 1)))
        r = np.sqrt(area / np.pi)
        self.r = r
        self.x = x
        self.xkm = xkm
        self.y = y
        self.ykm = ykm
        return (r >= rmin and r <= rmax)

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Calculates the distance between two points"""
        return _distance(x1, y1, x2, y2)

    def is_contour_not_circular(self, errmax=0.35):
        """Detects the circularity of a contour by comparing
        mean deviation from effective radius. Returns true if
        contour is not too distorted."""
        x = self.xkm
        y = self.ykm
        r = self.r
        centroidx = np.mean(x, axis=0)
        centroidy = np.mean(y, axis=0)
        self.xckm = centroidx
        self.yckm = centroidy
        self.xc = np.mean(self.ctr[:, 0])
        self.yc = np.mean(self.ctr[:, 1])
        xdist = x - centroidx
        ydist = y - centroidy
        rlocal = self.distance(x, y, centroidx, centroidy)
        area_error = np.mean(np.fabs(r**2 - rlocal**2))
        area_error = area_error / r**2
        return area_error <= errmax

    def is_vort_sign_homogeneous(self):
        """Finds if the sign of vorticity is
        same everywhere inside the contour."""
        rzeta = self.variables.get('rzeta')
        mask = self.in_hull(self.domain.coordsq, self.ctr).reshape(rzeta.shape)
        rzetainsidecontour = rzeta[mask]
        vortsame = (np.all(rzetainsidecontour < 0) if rzetainsidecontour[0] < 0
                    else np.all(rzetainsidecontour > 0))
        vortextreme = np.amax(np.fabs(rzetainsidecontour)) * np.sign(
            rzetainsidecontour[0])
        self.rzeta = vortextreme
        return vortsame

    def get_max_ssha(self):
        """Returns extreme SSHA inside contour"""
        e = self.variables.get('e')
        emean = self.mean_variables.get('emean')
        ssha = e - emean
        mask = self.in_hull(self.domain.coordsh, self.ctr).reshape(ssha.shape)
        sshainsidecontour = ssha[mask]
        sshamax = np.amax(sshainsidecontour)
        sshamin = np.amin(sshainsidecontour)
        if np.fabs(sshamin) > np.fabs(sshamax):
            self.ssha = sshamin
        else:
            self.ssha = sshamax

    @staticmethod
    def in_hull(p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p) >= 0


class EddyRegistry():
    """Class to hold eddy_lists. This mostly behaves like a list.
    It additionally has methods to assign eddies to tracks."""

    def __init__(self):
        self.erlist = []
        self._track_id = 0
        self._index_of_assigned_track_ids = 0
        self._first_time_step_assigned = False

    def __repr__(self):
        return """Eddy list holding lists of
        eddies from time = {} to {}.""".format(self[0].time, self[-1].time)

    def __getitem__(self, key):
        return self.erlist[key]

    #def __iter__(self):
    #    return iter(self.erlist)

    def __len__(self):
        return len(self.erlist)

    def iter_eddy(self):
        return iter([eddy for eddy_list in self.erlist for eddy in eddy_list])

    def append(self, eddy_list):
        assert isinstance(eddy_list, EddyListAtTime)
        self.erlist.append(eddy_list)
        self.erlist.sort(key=lambda eddy_list: eddy_list.time)
        self._assign_track_ids(eddy_list)

    def append_eddy(self, eddy):
        assert insinstance(eddy, Eddy)
        time = eddy.time
        for elist in self.erlist:
            if elist.time == time:
                elist.append(eddy)
                break
        else:
            new_list = EddyListAtTime(eddy.time)
            new_list.append(eddy)
            self.erlist.append(new_list)

    def _assign_track_ids(self, eddy_list):
        if len(self.erlist) == 1 or self._first_time_step_assigned is False:
            if self.erlist[0].time == 0:
                for eddy_new in eddy_list:
                    eddy_new.track_id = self._track_id
                    self._track_id += 1
                self._first_time_step_assigned = True
        else:
            for i in range(self._index_of_assigned_track_ids,
                           len(self.erlist) - 1):
                if i == self.erlist[i + 1].time - 1:
                    for eddy_new in self.erlist[i + 1]:
                        track_assigned = False
                        for eddy_old in self.erlist[i]:
                            if self._eddies_satisfy_track_conditions(
                                    eddy_new, eddy_old):
                                eddy_new.track_id = eddy_old.track_id
                                track_assigned = True
                                break
                        if not track_assigned:
                            eddy_new.track_id = self._track_id
                            self._track_id += 1
                    self._index_of_assigned_track_ids = i + 1

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Calculates the distance between two points"""
        return _distance(x1, y1, x2, y2)

    def _is_new_contour_close(self, eddy1, eddy2, rmultiplier=1.2):
        """Returns true if distance between centers of two eddies
        is less than the sum of their radii."""
        return (self.distance(eddy1.xckm, eddy1.yckm, eddy2.xckm, eddy2.yckm) <
                rmultiplier * (eddy1.r + eddy2.r))

    def _is_change_in_area_reasonable(self,
                                      eddy1,
                                      eddy2,
                                      min_ratio=0.25,
                                      max_ratio=2.5):
        """Returns true if the ratio of areas of two eddies is
        between min_ratio and max_ratio."""
        area_ratio = eddy1.r**2 / eddy2.r**2
        return (area_ratio >= min_ratio and area_ratio <= max_ratio)

    def _is_change_in_vort_reasonable(self,
                                      eddy1,
                                      eddy2,
                                      min_ratio=0.25,
                                      max_ratio=2.5):
        """Returns true if the ratio of vorticities of two eddies is
        between min_ratio and max_ratio."""
        vort_ratio = eddy1.rzeta / eddy2.rzeta
        return (vort_ratio >= min_ratio and vort_ratio <= max_ratio)

    def _does_vort_sign_match(self, eddy1, eddy2):
        """Returns true if vorticities of both eddies have
        same sign"""
        return np.sign(eddy1.rzeta) == np.sign(eddy2.rzeta)

    def _eddies_satisfy_track_conditions(self, eddy1, eddy2):
        """Returns true if two eddies satisfy track condtions."""
        return (self._is_new_contour_close(eddy1, eddy2)
                and self._is_change_in_area_reasonable(eddy1, eddy2)
                and self._is_change_in_vort_reasonable(eddy1, eddy2))


class EddyListAtTime():
    """A list to hold eddies at a particular time step"""

    def __init__(self, time):
        self.elist = []
        self.time = time

    def __repr__(self):
        return """List of eddies at time = {}.""".format(self.time)

    def __getitem__(self, key):
        return self.elist[key]

    #def __iter__(self):
    #    return iter(self.elist)

    def __len__(self):
        return len(self.elist)

    def append(self, eddy):
        if not self._eddy_already_detected(eddy):
            self.elist.append(eddy)

    def get(self, id_):
        for item in self.elist:
            if item.id_ == id_:
                return item
        raise IndexError('Eddy not found.')

    def _eddy_already_detected(self, eddy):
        """Returns True if eddy already detected."""
        assert isinstance(eddy, Eddy)
        eddy_already_present = False
        if len(self.elist) > 0:
            for detected_eddy in self.elist:
                if self._eddies_same(eddy, detected_eddy):
                    eddy_already_present = True
                    break
        return eddy_already_present

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Calculates the distance between two points"""
        return _distance(x1, y1, x2, y2)

    def _does_vort_sign_match(self, eddy1, eddy2):
        """Returns true if vorticities of both eddies have
        same sign"""
        return np.sign(eddy1.rzeta) == np.sign(eddy2.rzeta)

    def _is_center_inside_prev_eddy(self, eddy1, eddy2):
        """Returns true if the center of eddy1 is
        inside eddy2"""
        return (self.distance(eddy1.xckm, eddy1.yckm, eddy2.xckm, eddy2.yckm) <
                eddy2.r)

    def _eddies_same(self, eddy1, eddy2):
        """Returns true if both eddies are deemed the same."""
        return (self._does_vort_sign_match(eddy1, eddy2)
                and self._is_center_inside_prev_eddy(eddy1, eddy2))


class Domain():
    def __init__(self):
        """This class stores information about the geometry of the domain."""

        # It is best to read xh, yh, xq, yq, and tim from model
        # output files. If reading from a file, make sure to pass file
        # name and open it here.

        self.xq = None # Define the x locations of vorticity points
        self.yq = None # Define the y locations of vorticity points
        self.xh = None # Define the x locations of tracer points
        self.yh = None # Define the y locations of tracer points
        self.tim = None # Define the time steps of your data

        [xx, yy] = np.meshgrid(self.xh, self.yh)
        xx = xx.reshape(xx.size)
        yy = yy.reshape(yy.size)
        self.coordsh = np.vstack((xx, yy)).T

        [xx, yy] = np.meshgrid(self.xq, self.yq)
        xx = xx.reshape(xx.size)
        yy = yy.reshape(yy.size)
        self.coordsq = np.vstack((xx, yy)).T

def get_eddy(Domain, contour_levels, eddyfactory, lock, time_steps,
             mean_variables, z):
    """Determines the coordinates of contours of qparam at
             contour_levels, passes them on to tentative_eddy, and
             puts them in the queue, eddyfactory, if tentative_eddy
             is confirmed to be an actual eddy."""

    pname = multiprocessing.current_process().name
    while True:
        time = time_steps.get(False)
        if time is None:
            break
        print('{} assigned time step {}'.format(
            multiprocessing.current_process().name, time))

        variables = read_data(time, lock, z) # MODIFY ARGUMENTS ACCORDING TO YOUR IMPLEMENTATION
        wparam = variables.get('wparam')
        xx, yy = np.meshgrid(Domain.xh, Domain.yh)
        contour_field = QuadContourGenerator(
            xx, yy, wparam, None, mpl.rcParams['contour.corner_mask'], 0)

        eddy_list = EddyListAtTime(time)
        id_left = str(time) + str('_')
        id_right = 0
        for clevs in contour_levels:
            ctr_list = contour_field.create_contour(clevs)
            for i, ctr in enumerate(ctr_list):
                temp_eddy = TentativeEddy(ctr, time, variables, Domain,
                                          mean_variables)
                if temp_eddy.is_eddy:
                    eddy = Eddy(tentative_eddy=temp_eddy)
                    eddy.id_ = id_left + str(id_right)
                    eddy_list.append(eddy)
                    id_right += 1
        eddyfactory.put(eddy_list)
        print("""{} finished time step {}. Found {} eddies!""".format(
            pname, time, len(eddy_list)))
    print('Time steps exhausted! {} exiting!'.format(pname))


def read_mean_data():
    """This function returns mean SSH, which is stored as emean in the
    returned dictionary."""

    # ssh = MODIFY HERE (make sure that ssh is 2D numpy array and located at tracer points)

    assert ssh.ndim == 2
    return dict(emean=ssh)


def read_data(tim, lock, z):
    """This function returns OW parameter (wparam), relative vorticity
    (rzeta) both at height z, and SSH (e) at time tim."""

    # UNCOMMENT THE FOLLOWING LINE IF YOU ARE USING NETCDF4 TO READ
    # DATA FROM FILE. THIS IS TO PREVENT TWO PROCESSES FROM ACCESSING
    # A FILE AT THE SAME TIME.
    # lock.acquire()

    # wparam = Make sure it is 2D numpy array and located at tracer points
    # rzeta  = Make sure it is 2D numpy array and located at vorticity points
    # SSH    = Make sure it is 2D numpy array and located at tracer points

    # UNCOMMENT THE FOLLOWING LINE IF YOU ARE USING NETCDF4 TO READ DATA FROM FILE
    # lock.release()

    assert wparam.ndim == 2
    assert rzeta.ndim == 2
    assert SSH.ndim == 2
    return dict(wparam=wparam, rzeta=rzeta, e=SSH)


def find_eddies(Domain,
                process_count,
                z,
                contour_levels):
    "Delegates eddy tracking responsibilities to multiple processes"

    lock = multiprocessing.RLock()
    tsteps = Domain.tim.size
    eddyfactory = multiprocessing.Manager().Queue(tsteps)
    print('Time steps to be processed is {}.'.format(tsteps))

    time_steps = multiprocessing.Queue(tsteps + process_count)
    for i in range(tsteps):
        time_steps.put(i)
    for i in range(process_count):
        time_steps.put(None)

    st = time.time()
    mean_variables = read_mean_data() # MODIFY ARGUMENTS ACCORDING TO YOUR IMPLEMENTATION
    jobs = []
    for i in range(process_count):
        p = multiprocessing.Process(
            target=get_eddy,
            args=(Domain, contour_levels, eddyfactory, lock, time_steps,
                  mean_variables, z))
        p.start()
        jobs.append(p)

    print('Accessing queue...')
    eddies = EddyRegistry()
    for i in range(tsteps):
        print('Waiting for {} list.'.format(i))
        eddy = eddyfactory.get()
        print('{} list received!'.format(eddy.time))
        eddies.append(eddy)
        print('{} list appended!'.format(eddy.time))
        print('Objects still in queue: {}'.format(eddyfactory.qsize()))
    print('All timesteps read! Active processes: {}'.format(
        multiprocessing.active_children()))

    for p in jobs:
        p.join(10)
        print(p.name, p.exitcode)

    print('All processes joined!')

    print('Total time taken: {}s'.format(time.time() - st))
    return eddies


def main(pickle_file, process_count=48, z=-1,
         vmin=-10.3011, vmax=-8.6989, clevs=20):
    """This function is called from the python interpreter to run the
         eddy tracking program.

    :param pickle_file: Eddies and tracks will be pickled in this file
    :param process_count: Number of processor cores the program will
    use (should be <= max number of cores given by
    multiprocessing.cpu_count() for efficiency )
    :param z: The height at which eddy tracking is done (implement
    this according to your need)
    :param vmin: Max value of OW threshold at which search for eddies
    is carried out
    :param vmax: Min value of OW threshold at which search for eddies
    is carried out
    :param clevs: Number of levels between vmin and vmax separated on
    a log scale (this is ignored if vmax == vmin)
    :returns: None

    """
    d = Domain() # ADD ARGUMENTS TO THIS CALL IF YOUR IMPLEMENTATION NEEDS THEM

    if vmax == vmin:
        contour_levels = [vmax]
    else:
        contour_levels = -np.logspace(vmin, vmax, clevs)
    eddies = find_eddies(d, process_count, z, contour_levels)
    print('Received eddy registry!')
    tracks = groupby(lambda eddy: eddy.track_id, eddies.iter_eddy())
    with open(pickle_file, mode='wb') as f:
        print('Dumping data!')
        pickle.dump((eddies, tracks), f)
        print('Done!')
