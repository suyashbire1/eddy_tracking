import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._cntr as cntr
from netCDF4 import Dataset as dset, MFDataset as mfdset
from scipy.spatial import Delaunay
import time
import pickle
import seaborn.apionly as sns
from toolz import groupby,partial
import multiprocessing
import pyximport
pyximport.install()
import getvaratzc as gvz
import queue

class Eddy:
    """A class to hold information about detected eddies"""

#    def __init__(self,identifier,ctr,x,y,xkm,ykm,r,rzeta,rzetac,e,ec,ssha,sshac,t):
    def __init__(self,*args,**kwargs):
        """Initializes an eddy track"""
        tentative_eddy = kwargs.get('tentative_eddy',None)
        if tentative_eddy is None:
            for k,v in kwargs.items:
                setattr(self,k,v)
        else:
            self.init_from_tentative(tentative_eddy)
        self._id_ = None
        self._track_id = None

    def __repr__(self):
        return 'Eddy {} at time {}'.format(self._id_,self.time)

    def init_from_tentative(self,tentative_eddy):
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
    def id_(self,id_):
        self._id_ = id_

    @property
    def track_id(self):
        return self._track_id

    @track_id.setter
    def track_id(self,track_id):
        self._track_id = track_id


class TentativeEddy:
    """A class to hold a contour and methods to check if the
    contour contains eddies."""
    def __init__(self,ctr,time,variables,domain,mean_variables):
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
        return not (ctr.view(dtype).shape[0] ==
                    np.unique(ctr.view(dtype)).shape[0])

    @staticmethod
    def convert_contour_to_km(ctr):
        """Converts vertices from degree to km"""
        R = 6378
        x, y = ctr[:,0], ctr[:,1]
        ykm = R*np.radians(y)
        xkm = R*np.cos(np.radians(y))*np.radians(x)
        return x,y,xkm,ykm

    def is_radius_too_extreme(self,rmin=15,rmax=150):
        """Retruns the effective radius, a boolean
        which is true if radius of contour is within
        reasonable range (default 15 to 150 km), and
        coordinates of the contour in km"""
        x,y,xkm,ykm = self.convert_contour_to_km(self.ctr)
        area = 0.5*np.fabs(np.dot(xkm,np.roll(ykm,-1)-np.roll(ykm,1)))
        r = np.sqrt(area/np.pi)
        self.r = r
        self.x = x
        self.xkm = xkm
        self.y = y
        self.ykm = ykm
        return (r >= rmin and r <= rmax)

    @staticmethod
    def distance(x1,y1,x2,y2):
        """Calculates the distance between two points"""
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    def is_contour_not_circular(self,errmax=0.35):
        """Detects the circularity of a contour by comparing
        mean deviation from effective radius. Returns true if
        contour is not too distorted."""
        x = self.xkm
        y = self.ykm
        r = self.r
        centroidx = np.mean(x,axis=0)
        centroidy = np.mean(y,axis=0)
        self.xckm = centroidx
        self.yckm = centroidy
        self.xc = np.mean(self.ctr[:,0])
        self.yc = np.mean(self.ctr[:,1])
        xdist = x-centroidx
        ydist = y-centroidy
        rlocal = self.distance(x,y,centroidx,centroidy)
        area_error = np.mean(np.fabs(r**2 - rlocal**2))
        area_error = area_error/r**2
        return area_error <= errmax

    def is_vort_sign_homogeneous(self):
        """Finds if the sign of vorticity is
        same everywhere inside the contour."""
        rzeta = self.variables.get('rzeta')
        mask = self.in_hull(self.domain.coordsh,self.ctr).reshape(rzeta.shape)
        rzetainsidecontour = rzeta[mask]
        vortsame = (np.all(rzetainsidecontour < 0) if rzetainsidecontour[0] < 0
                else np.all(rzetainsidecontour > 0))
        vortextreme = np.amax(np.fabs(rzetainsidecontour))*np.sign(rzetainsidecontour[0])
        self.rzeta = vortextreme
        return vortsame

    def get_max_ssha(self):
        """Finds if the sign of vorticity is
        same everywhere inside the contour."""
        e = self.variables.get('e')
        emean = self.mean_variables.get('emean')
        ssha = e - emean
        mask = self.in_hull(self.domain.coordsh,self.ctr).reshape(ssha.shape)
        sshainsidecontour = ssha[mask]
        sshamax = np.amax(sshainsidecontour)
        sshamin = np.amin(sshainsidecontour)
        if np.fabs(sshamin) > np.fabs(sshamax):
            self.ssha = sshamin
        else:
            self.ssha = sshamax

    @staticmethod
    def in_hull(p,hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p)>=0


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
        eddies from time = {} to {}.""".format(self[0].time,self[-1].time)

    def __getitem__(self,key):
        return self.erlist[key]

    #def __iter__(self):
    #    return iter(self.erlist)

    def __len__(self):
        return len(self.erlist)

    def iter_eddy(self):
        return iter([eddy for eddy_list in self.erlist for eddy in eddy_list])

    def append(self,eddy_list):
        assert isinstance(eddy_list, EddyListAtTime)
        self.erlist.append(eddy_list)
        self.erlist.sort(key=lambda eddy_list:eddy_list.time)
        self._assign_track_ids(eddy_list)

    def _assign_track_ids(self,eddy_list):
        if len(self.erlist) == 1 or self._first_time_step_assigned is False:
            if self.erlist[0].time == 0:
                for eddy_new in eddy_list:
                    eddy_new.track_id = self._track_id
                    self._track_id += 1
                self._first_time_step_assigned = True
        else:
            for i in range(self._index_of_assigned_track_ids,len(self.erlist)-1):
                if i == self.erlist[i+1].time - 1:
                    for eddy_new in self.erlist[i+1]:
                        track_assigned = False
                        for eddy_old in self.erlist[i]:
                            if self._eddies_satisfy_track_conditions(eddy_new,eddy_old):
                                eddy_new.track_id = eddy_old.track_id
                                track_assigned = True
                                break
                        if not track_assigned:
                            eddy_new.track_id = self._track_id
                            self._track_id += 1
                    self._index_of_assigned_track_ids = i+1

    def _distance(self,x1,y1,x2,y2):
        """Calculates the distance between two points"""
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    def _is_new_contour_close(self,eddy1,eddy2,rmultiplier=1.2):
        """Returns true if distance between centers of two eddies
        is less than the sum of their radii."""
        return (self._distance(eddy1.xckm,eddy1.yckm,
                               eddy2.xckm,eddy2.yckm) <
                rmultiplier*(eddy1.r + eddy2.r))

    def _is_change_in_area_reasonable(self,eddy1,eddy2,
                                     min_ratio=0.25,max_ratio=2.5):
        """Returns true if the ratio of areas of two eddies is
        between min_ratio and max_ratio."""
        area_ratio = eddy1.r**2 / eddy2.r**2
        return (area_ratio >= min_ratio and area_ratio <= max_ratio)

    def _is_change_in_vort_reasonable(self,eddy1,eddy2,
                                     min_ratio=0.25,max_ratio=2.5):
        """Returns true if the ratio of vorticities of two eddies is
        between min_ratio and max_ratio."""
        vort_ratio = eddy1.rzeta / eddy2.rzeta
        return (vort_ratio >= min_ratio and vort_ratio <= max_ratio)

    def _does_vort_sign_match(self,eddy1,eddy2):
        """Returns true if vorticities of both eddies have
        same sign"""
        return np.sign(eddy1.rzeta) == np.sign(eddy2.rzeta)

    def _eddies_satisfy_track_conditions(self,eddy1,eddy2):
        """Returns true if two eddies satisfy track condtions."""
        return(    self._is_new_contour_close(eddy1,eddy2)
               and self._is_change_in_area_reasonable(eddy1,eddy2)
               and self._is_change_in_vort_reasonable(eddy1,eddy2))


class EddyListAtTime():
    """A list to hold eddies at a particular time step"""
    def __init__(self,time):
        self.elist = []
        self.time = time

    def __repr__(self):
        return """List of eddies at time = {}.""".format(self.time)

    def __getitem__(self,key):
        return self.elist[key]

    #def __iter__(self):
    #    return iter(self.elist)

    def __len__(self):
        return len(self.elist)

    def append(self,eddy):
        if not self._eddy_already_detected(eddy):
            self.elist.append(eddy)

    def get(self,id_):
        for item in self.elist:
            if item.id_ == id_:
                return item
        raise IndexError('Eddy not found.')

    def _eddy_already_detected(self,eddy):
        """Returns True if eddy already detected."""
        assert isinstance(eddy,Eddy)
        eddy_already_present = False
        if len(self.elist) > 0:
            for detected_eddy in self.elist:
                if self._eddies_same(eddy,detected_eddy):
                    eddy_already_present = True
                    break
        return eddy_already_present

    def _distance(self,x1,y1,x2,y2):
        """Calculates the distance between two points"""
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    def _does_vort_sign_match(self,eddy1,eddy2):
        """Returns true if vorticities of both eddies have
        same sign"""
        return np.sign(eddy1.rzeta) == np.sign(eddy2.rzeta)

    def _is_center_inside_prev_eddy(self,eddy1,eddy2):
        """Returns true if the center of eddy1 is
        inside eddy2"""
        return (self._distance(eddy1.xckm,eddy1.yckm,
                               eddy2.xckm,eddy2.yckm) < eddy2.r)

    def _eddies_same(self,eddy1,eddy2):
        """Returns true if both eddies are deemed the same."""
        return (self._does_vort_sign_match(eddy1,eddy2) and
                self._is_center_inside_prev_eddy(eddy1,eddy2))


class Domain():
    def __init__(self,fil,geofil,vgeofil):
        """Initialize the time invariant quantities."""

        self.fhg = dset(geofil)
        self.fhgv = dset(vgeofil)
        self.fh = mfdset(fil)
        self.fhgv = self.fhg.variables
        self.fhv = self.fh.variables

        self.xq = self.fhv['xq'][:]
        self.yq = self.fhv['yq'][:]
        self.xh = self.fhv['xh'][:]
        self.yh = self.fhv['yh'][:]
        self.zi = self.fhv['zi'][:]
        self.zl = self.fhv['zl'][:]
        self.tim = self.fhv['Time'][:]
        self.dxbu = self.fhgv['dxBu'][:]
        self.dxcv = self.fhgv['dxCv'][:]
        self.dycv = self.fhgv['dyCv'][:]
        self.dybu = self.fhgv['dyBu'][:]
        self.dxt = self.fhgv['dxT'][:]
        self.dyt = self.fhgv['dyT'][:]
        [xx,yy] = np.meshgrid(self.xq,self.yq)
        xx = xx.reshape(xx.size)
        yy = yy.reshape(yy.size)
        self.coordsq = np.vstack((xx,yy)).T
        [xx,yy] = np.meshgrid(self.xh,self.yh)
        self.R = 6.378e3 #km
        self.omega = 2*np.pi*(1/24/3600 + 1/365/24/3600)
        self.fhh = 2*self.omega*np.sin(np.radians(yy))
        xx = xx.reshape(xx.size)
        yy = yy.reshape(yy.size)
        self.coordsh = np.vstack((xx,yy)).T
        self.fq = self.fhgv['f'][:]

    def close(self):
        self.fhg.close()
        self.fhgv.close()
        self.fh.close()


def get_eddy(Domain,contour_levels,eddyfactory,lock,time_steps,mean_variables):
    """Returns coordinates of contour of qparam at clev"""

    while True:
        try:
            time = time_steps.get()
            print('{} assigned time step {}'.format(multiprocessing.current_process().name, time))
            #lock.acquire()
            variables = read_data(Domain,time,lock)
            #lock.release()
            wparam = variables.get('wparam')
            xx, yy = np.meshgrid(Domain.xh, Domain.yh)
            c = cntr.Cntr(xx,yy,wparam)

            eddy_list = EddyListAtTime(time)
            id_left = str(time) + str('_')
            id_right = 0
            for clevs in contour_levels:
                nlist = c.trace(clevs,clevs,0)
                ctr_list = nlist[:len(nlist)//2]
                for i,ctr in enumerate(ctr_list):
                    temp_eddy = TentativeEddy(ctr,time,variables,Domain,mean_variables)
                    if temp_eddy.is_eddy:
                        eddy = Eddy(tentative_eddy=temp_eddy)
                        eddy.id_ = id_left + str(id_right)
                        eddy_list.append(eddy)
                        id_right += 1
            eddyfactory.put(eddy_list)
            print("""{} finished time step {}. Found {} eddies!""".format(multiprocessing.current_process().name,time,len(eddy_list)))
        except queue.Empty:
            eddyfactory.close()
            eddyfactory.join_thread()
            break

def read_data1(fil,time):
    fh = mfdset(fil)
    from pym6 import Variable, Domain
    gv = Variable.GridVariable
    Domain = Domain.Domain
    d = Domain(geofil, vgeofil,-25,0,10,60,ts=time,te=time+1)
    e = gv('e',d,'hl',fh).read_array(tmean=False)
    wparam = gv('wparam',d,'hl',fh).read_array(tmean=False).toz(-1,e=e).values.squeeze()
    uy = (gv('u',d,'ul',fh).xsm().ysm().yep()
          .read_array(extend_kwargs=dict(method='zeros'),tmean=False,filled=0)
          .move_to('hl').move_to('vl').ddx(2).toz(-1,e=e).values.squeeze())
    vx = (gv('v',d,'vl',fh).ysm().xsm().xep()
          .read_array(extend_kwargs=dict(method='zeros'),tmean=False,filled=0)
          .move_to('hl').move_to('ul').ddx(3).toz(-1,e=e).values.squeeze())
    print(np.max(uy),np.min(uy),np.max(vx),np.min(vx))
    print(np.all(uy==0),np.all(vx==0))
    zeta = vx - uy
    rzeta = zeta/d.f
    #plt.pcolormesh(d.lonh,d.lath,wparam,vmax=1e-8,vmin=-1e-8,cmap='RdBu_r')
    plt.pcolormesh(d.lonh,d.lath,zeta,vmax=1e-5,vmin=-1e-5,cmap='RdBu_r')
    plt.colorbar()

def read_mean_data(Domain):
    e = Domain.fhv['e'][:,0,:,:]
    e = np.mean(e,axis=0)
    return dict(emean = e)

def read_data(Domain,time,lock):
    lock.acquire()
    wparam = Domain.fhv['wparam'][time:time+1,:,:,:]
    e = Domain.fhv['e'][time:time+1,:,:,:]
    u = Domain.fhv['u'][time:time+1,:,:,:].filled(0)
    v = Domain.fhv['v'][time:time+1,:,:,:].filled(0)
    lock.release()

    u = np.concatenate((np.zeros(u[:,:,:,:1].shape),u),axis=3)
    u = 0.5*(u[:,:,:,:-1] + u[:,:,:,1:])
    u = np.concatenate((-u[:,:,:1,:],u,-u[:,:,-1:,:]),axis=2)
    u = 0.5*(u[:,:,:-1,:] + u[:,:,1:,:])
    uy = np.diff(u,axis=2)/Domain.dyt

    v = np.concatenate((np.zeros(v[:,:,:1,:].shape),v),axis=2)
    v = 0.5*(v[:,:,:-1,:] + v[:,:,1:,:])
    v = np.concatenate((-v[:,:,:,:1],v,-v[:,:,:,-1:]),axis=3)
    v = 0.5*(v[:,:,:,:-1] + v[:,:,:,1:])
    vx = np.diff(v,axis=3)/Domain.dxt

    zeta = vx - uy
    rzeta = zeta/Domain.fhh

    wparam = gvz.getvaratzc(wparam,np.array([-1], dtype=np.float32),e)
    rzeta = gvz.getvaratzc(rzeta.astype(np.float32),np.array([-1], dtype=np.float32),e)
    #plt.pcolormesh(Domain.xh,Domain.yh,wparam[0,0,:,:],vmax=1e-8,vmin=-1e-8,cmap='RdBu_r')
    #plt.pcolormesh(d.lonh,d.lath,rzeta,vmax=1,vmin=-1,cmap='RdBu_r')
    #plt.colorbar()
    return dict(wparam=wparam[0,0,:,:],rzeta=rzeta[0,0,:,:],e=e[0,0,:,:])


def find_eddies(Domain,vmin=-10.3,vmax=-8.8):
    """Searches contours at regular intervals between
    vmin and vmax and checks if they are valid eddies"""

    contour_levels = np.sort(-4*np.logspace(vmin,vmax))
    lock = multiprocessing.RLock()
    tsteps = Domain.tim.size
    eddyfactory = multiprocessing.JoinableQueue(tsteps)
    print(tsteps)

    time_steps = multiprocessing.Queue(tsteps)
    for i in range(tsteps):
        time_steps.put(i)
#    time_steps.close() # No more data will be added to time_steps
#    time_steps.join_thread() # Wait till all the data has been flushed to time_steps

    st = time.time()
    mean_variables = read_mean_data(Domain)
    process_count = multiprocessing.cpu_count() - 1
    #process_count=6
    jobs = []
    for i in range(process_count):
        p = multiprocessing.Process(target=get_eddy,args=(Domain,contour_levels,eddyfactory,
                                                          lock,time_steps,mean_variables))
        p.start()
        jobs.append(p)

    print('Accessing queue...')
    eddies = EddyRegistry()
    for i in range(tsteps):
        eddies.append(eddyfactory.get(True))
        eddyfactory.task_done()
    eddyfactory.join()

    for j in jobs:
        j.join()

    print('Total time taken: {}s'.format(time.time()-st))
    return eddies

def main(start,end,pickle_file):
    fil = ['output__00{}.nc'.format(n) for n in range(start,end)]
    geofil = 'ocean_geometry.nc'
    vgeofil = 'Vertical_coordinate.nc'
    d = Domain(fil,geofil,vgeofil)
    a = find_eddies(d)
    d.close()
    #tracks = groupby(lambda eddy:eddy.track_id,eddies.iter_eddy())
    pickle.dump(a,open(pickle_file,mode='wb'))
