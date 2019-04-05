"""
Streamline plotting for 3D vector fields.

"""

import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


__all__ = ['streamplot3D']


def streamplot3D(axes, x, y, z, u, v, w, density=1, linewidth=None, color=None,
               cmap=None, norm=None, arrowsize=1, arrowstyle='-|>',
               minlength=0.1, transform=None, zorder=None, start_points=None,
               maxlength=4.0, integration_direction='both'):
    """
    Draw streamlines of a vector flow.

    Parameters
    ----------
    x, y , z : 1d arrays
        An evenly spaced grid.
    u, v , w : 3d arrays
        *x* and *y*-velocities. Number of rows should match length of *y*, and
        the number of columns should match *x*.
    density : float or 2-tuple
        Controls the closeness of streamlines. When ``density = 1``, the domain
        is divided into a 30x30 grid---*density* linearly scales this grid.
        Each cell in the grid can have, at most, one traversing streamline.
        For different densities in each direction, use [density_x, density_y].
    linewidth : numeric or 2d array
        Vary linewidth when given a 2d array with the same shape as velocities.
    color : matplotlib color code, or 2d array
        Streamline color. When given an array with the same shape as
        velocities, *color* values are converted to colors using *cmap*.
    cmap : `~matplotlib.colors.Colormap`
        Colormap used to plot streamlines and arrows. Only necessary when using
        an array input for *color*.
    norm : `~matplotlib.colors.Normalize`
        Normalize object used to scale luminance data to 0, 1. If ``None``,
        stretch (min, max) to (0, 1). Only necessary when *color* is an array.
    arrowsize : float
        Factor scale arrow size.
    arrowstyle : str
        Arrow style specification.
        See `~matplotlib.patches.FancyArrowPatch`.
    minlength : float
        Minimum length of streamline in axes coordinates.
    start_points : Nx3 array
        Coordinates of starting points for the streamlines.
        In data coordinates, the same as the *x*, *y* and *z* arrays.
    zorder : int
        Any number.
    maxlength : float
        Maximum length of streamline in axes coordinates.
    integration_direction : ['forward' | 'backward' | 'both']
        Integrate the streamline in forward, backward or both directions.
        default is ``'both'``.

    Returns
    -------
    stream_container : StreamplotSet
        Container object with attributes

        - lines: `matplotlib.collections.LineCollection` of streamlines

        - arrows: collection of `matplotlib.patches.FancyArrowPatch`
          objects representing arrows half-way along stream
          lines.

        This container will probably change in the future to allow changes
        to the colormap, alpha, etc. for both lines and arrows, but these
        changes should be backward compatible.
    """
    grid = Grid(x, y, z)
    mask = StreamMask(density)
    dmap = DomainMap(grid, mask)

    if zorder is None:
        zorder = mlines.Line2D.zorder

    # default to data coordinates
    if transform is None:
        transform = axes.transData

    if color is None:
        color = axes._get_lines.get_next_color()

    if linewidth is None:
        linewidth = matplotlib.rcParams['lines.linewidth']

    line_kw = {}
    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)

    if integration_direction not in ['both', 'forward', 'backward']:
        errstr = ("Integration direction '%s' not recognised. "
                  "Expected 'both', 'forward' or 'backward'." %
                  integration_direction)
        raise ValueError(errstr)

    if integration_direction == 'both':
        maxlength /= 2.

    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        if color.shape != grid.shape:
            raise ValueError(
                "If 'color' is given, must have the shape of 'Grid(x,y)'")
        line_colors = []
        color = np.ma.masked_invalid(color)
    else:
        line_kw['color'] = color
        arrow_kw['color'] = color

    if isinstance(linewidth, np.ndarray):
        if linewidth.shape != grid.shape:
            raise ValueError(
                "If 'linewidth' is given, must have the shape of 'Grid(x,y)'")
        line_kw['linewidth'] = []
    else:
        line_kw['linewidth'] = linewidth
        arrow_kw['linewidth'] = linewidth

    line_kw['zorder'] = zorder
    arrow_kw['zorder'] = zorder

    ## Sanity checks.
    if u.shape != grid.shape or v.shape != grid.shape or w.shape !=grid.shape:
        raise ValueError("'u' ,'v' and 'w' must be of shape 'Grid(x,y,z)'")

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)
    w = np.ma.masked_invalid(w)

    integrate = get_integrator(u, v, w, dmap, minlength, maxlength,
                               integration_direction)

    trajectories = []
    if start_points is None:
        for xm, ym, zm in _gen_starting_points(mask.shape):
            # print([xm, ym, zm]) Check starting point generation
            if mask[ym, xm, zm] == 0:
                xg, yg, zg = dmap.mask2grid(xm, ym, zm)
                t = integrate(xg, yg, zg)
                if t is not None:
                    trajectories.append(t)
    else:
        sp2 = np.asanyarray(start_points, dtype=float).copy()
        # Check if start_points are outside the data boundaries
        for xs, ys, zs in sp2:
            if not (grid.x_origin <= xs <= grid.x_origin + grid.width
                    and grid.y_origin <= ys <= grid.y_origin + grid.height
                    and grid.z_origin <= zs <= grid.z_origin + grid.length):
                raise ValueError("Starting point ({}, {}, {}) outside of data "
                                 "boundaries".format(xs, ys, zs))

        # Convert start_points from data to array coords
        # Shift the seed points from the bottom left of the data so that
        # data2grid works properly.
        sp2[:, 0] -= grid.x_origin
        sp2[:, 1] -= grid.y_origin
        sp2[:, 2] -= grid.z_origin

        for xs, ys, zs in sp2:
            xg, yg, zg = dmap.data2grid(xs, ys, zs)
            t = integrate(xg, yg, zg)
            if t is not None:
                trajectories.append(t)

    if use_multicolor_lines:
        if norm is None:
            norm = mcolors.Normalize(color.min(), color.max())
        if cmap is None:
            cmap = cm.get_cmap(matplotlib.rcParams['image.cmap'])
        else:
            cmap = cm.get_cmap(cmap)

    streamlines = []
    arrows = []
    for t in trajectories:
        tgx = np.array(t[0])
        tgy = np.array(t[1])
        tgz = np.array(t[2])
        # Rescale from grid-coordinates to data-coordinates.
        tx, ty, tz = dmap.grid2data(*np.array(t))
        tx += grid.x_origin
        ty += grid.y_origin
        tz += grid.z_origin

        points = np.transpose([tx, ty, tz]).reshape(-1, 1, 3)
        streamlines.extend(np.hstack([points[:-1], points[1:]]))

        # Add arrows half way along each trajectory.
        s = np.cumsum(np.sqrt(np.diff(tx)**2+np.diff(ty)**2+np.diff(tz)**2))
        n = np.searchsorted(s, s[-1] / 2.)
        arrow_tail = (tx[n], ty[n], tz[n])
        arrow_head = (np.mean(tx[n:n + 2]), np.mean(ty[n:n + 2]), np.mean(tz[n:n + 2]))

        if isinstance(linewidth, np.ndarray):
            line_widths = interpgrid(linewidth, tgx, tgy, tgz)[:-1]
            line_kw['linewidth'].extend(line_widths)
            arrow_kw['linewidth'] = line_widths[n]

        if use_multicolor_lines:
            color_values = interpgrid(color, tgx, tgy, tgz)[:-1]
            line_colors.append(color_values)
            arrow_kw['color'] = cmap(norm(color_values[n]))

        p = Arrow3D([arrow_tail[0], arrow_head[0]], [arrow_tail[1], arrow_head[1]], [arrow_tail[2], arrow_head[2]], transform=transform, **arrow_kw)

        axes.add_artist(p)
        arrows.append(p)

    lc = Line3DCollection(streamlines, transform=transform, **line_kw)
    if use_multicolor_lines:
        lc.set_array(np.ma.hstack(line_colors))
        lc.set_cmap(cmap)
        lc.set_norm(norm)

    axes.add_collection3d(lc)
    axes.autoscale_view()

    ac = matplotlib.collections.PatchCollection(arrows)
    stream_container = StreamplotSet(lc, ac)
    return axes


class StreamplotSet(object):

    def __init__(self, lines, arrows, **kwargs):
        self.lines = lines
        self.arrows = arrows


# Coordinate definitions
# ========================

class DomainMap(object):
    """Map representing different coordinate systems.

    Coordinate definitions:

    * axes-coordinates goes from 0 to 1 in the domain.
    * data-coordinates are specified by the input x-y coordinates.
    * grid-coordinates goes from 0 to N and 0 to M for an N x M grid,
      where N and M match the shape of the input data.
    * mask-coordinates goes from 0 to N and 0 to M for an N x M mask,
      where N and M are user-specified to control the density of streamlines.

    This class also has methods for adding trajectories to the StreamMask.
    Before adding a trajectory, run `start_trajectory` to keep track of regions
    crossed by a given trajectory. Later, if you decide the trajectory is bad
    (e.g., if the trajectory is very short) just call `undo_trajectory`.
    """

    def __init__(self, grid, mask):
        self.grid = grid
        self.mask = mask
        # Constants for conversion between grid- and mask-coordinates
        self.x_grid2mask = (mask.nx - 1) / grid.nx
        self.y_grid2mask = (mask.ny - 1) / grid.ny
        self.z_grid2mask = (mask.nz - 1) / grid.nz

        self.x_mask2grid = 1. / self.x_grid2mask
        self.y_mask2grid = 1. / self.y_grid2mask
        self.z_mask2grid = 1. / self.z_grid2mask

        self.x_data2grid = 1. / grid.dx
        self.y_data2grid = 1. / grid.dy
        self.z_data2grid = 1. / grid.dz

    def grid2mask(self, xi, yi, zi):
        """Return nearest space in mask-coords from given grid-coords."""
        return (int(xi * self.x_grid2mask + 0.5),
                int(yi * self.y_grid2mask + 0.5),
                int(zi * self.z_grid2mask + 0.5))

    def mask2grid(self, xm, ym, zm):
        return xm * self.x_mask2grid, ym * self.y_mask2grid, zm * self.z_mask2grid

    def data2grid(self, xd, yd, zd):
        return xd * self.x_data2grid, yd * self.y_data2grid, zd * self.z_data2grid

    def grid2data(self, xg, yg, zg):
        return xg / self.x_data2grid, yg / self.y_data2grid, zg / self.z_data2grid

    def start_trajectory(self, xg, yg, zg):
        xm, ym, zm = self.grid2mask(xg, yg, zg)
        self.mask._start_trajectory(xm, ym, zm)

    def reset_start_point(self, xg, yg, zg):
        xm, ym, zm = self.grid2mask(xg, yg, zg)
        self.mask._current_xyz = (xm, ym, zm)

    def update_trajectory(self, xg, yg, zg):
        if not self.grid.within_grid(xg, yg, zg):
            raise InvalidIndexError
        xm, ym, zm = self.grid2mask(xg, yg, zg)
        self.mask._update_trajectory(xm, ym, zm)

    def undo_trajectory(self):
        self.mask._undo_trajectory()


class Grid(object):
    """Grid of data."""
    def __init__(self, x, y, z):

        if x.ndim == 1:
            pass
        elif x.ndim == 2:
            x_row = x[0, :]
            if not np.allclose(x_row, x):
                raise ValueError("The rows of 'x' must be equal")
            x = x_row
        else:
            raise ValueError("'x' can have at maximum 2 dimensions")

        if y.ndim == 1:
            pass
        elif y.ndim == 2:
            y_col = y[:, 0]
            if not np.allclose(y_col, y.T):
                raise ValueError("The columns of 'y' must be equal")
            y = y_col
        else:
            raise ValueError("'y' can have at maximum 2 dimensions")

        if z.ndim == 1:
            pass
        else:
            raise ValueError("'z' can have a maximum 1 dimesions")

        self.nx = len(x)
        self.ny = len(y)
        self.nz = len(z)

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0]

        self.x_origin = x[0]
        self.y_origin = y[0]
        self.z_origin = z[0]

        self.width = x[-1] - x[0]
        self.height = y[-1] - y[0]
        self.length = z[-1] - z[0]

        if not np.allclose(np.diff(x), self.width / (self.nx - 1)):
            raise ValueError("'x' values must be equally spaced")
        if not np.allclose(np.diff(y), self.height / (self.ny - 1)):
            raise ValueError("'y' values must be equally spaced")

    @property
    def shape(self):
        return self.ny, self.nx, self.nz

    def within_grid(self, xi, yi, zi):
        """Return True if point is a valid index of grid."""
        # Note that xi/yi can be floats; so, for example, we can't simply check
        # `xi < self.nx` since `xi` can be `self.nx - 1 < xi < self.nx`
        return xi >= 0 and xi <= self.nx - 1 and yi >= 0 and yi <= self.ny - 1 and zi >=0 and zi <=self.nz - 1


class StreamMask(object):
    """Mask to keep track of discrete regions crossed by streamlines.

    The resolution of this grid determines the approximate spacing between
    trajectories. Streamlines are only allowed to pass through zeroed cells:
    When a streamline enters a cell, that cell is set to 1, and no new
    streamlines are allowed to enter.
    """

    def __init__(self, density):
        try:
            self.nx, self.ny, self.nz = (30 * np.broadcast_to(density, 3)).astype(int)
        except ValueError:
            raise ValueError("'density' must be a scalar or be of length 3")
        if self.nx < 0 or self.ny < 0 or self.nz < 0:
            raise ValueError("'density' must be positive")
        self._mask = np.zeros((self.ny, self.nx, self.nz))
        self.shape = self._mask.shape

        self._current_xyz = None

    def __getitem__(self, *args):
        return self._mask.__getitem__(*args)

    def _start_trajectory(self, xm, ym, zm):
        """Start recording streamline trajectory"""
        self._traj = []
        self._update_trajectory(xm, ym, zm)

    def _undo_trajectory(self):
        """Remove current trajectory from mask"""
        for t in self._traj:
            self._mask.__setitem__(t, 0)

    def _update_trajectory(self, xm, ym, zm):
        """Update current trajectory position in mask.

        If the new position has already been filled, raise `InvalidIndexError`.
        """
        if self._current_xyz != (xm, ym, zm):
            if self[ym, xm, zm] == 0:
                self._traj.append((ym, xm, zm))
                self._mask[ym, xm, zm] = 1
                self._current_xyz = (xm, ym, zm)
            else:
                raise InvalidIndexError


class InvalidIndexError(Exception):
    pass


class TerminateTrajectory(Exception):
    pass


# Integrator definitions
#========================

def get_integrator(u, v, w, dmap, minlength, maxlength, integration_direction):

    # rescale velocity onto grid-coordinates for integrations.
    u, v, w = dmap.data2grid(u, v, w)

    # speed (path length) will be in axes-coordinates
    u_ax = u / dmap.grid.nx
    v_ax = v / dmap.grid.ny
    w_ax = w / dmap.grid.nz
    speed = np.ma.sqrt(u_ax ** 2 + v_ax ** 2 + w_ax ** 2)

    def forward_time(xi, yi, zi):
        ds_dt = interpgrid(speed, xi, yi, zi)
        if ds_dt == 0:
            raise TerminateTrajectory()
        dt_ds = 1. / ds_dt
        ui = interpgrid(u, xi, yi, zi)
        vi = interpgrid(v, xi, yi, zi)
        wi = interpgrid(w, xi, yi, zi)
        return ui * dt_ds, vi * dt_ds, wi * dt_ds

    def backward_time(xi, yi, zi):
        dxi, dyi, dzi = forward_time(xi, yi, zi)
        return -dxi, -dyi, -dzi

    def integrate(x0, y0, z0):
        """Return x, y, z grid-coordinates of trajectory based on starting point.

        Integrate both forward and backward in time from starting point in
        grid coordinates.

        Integration is terminated when a trajectory reaches a domain boundary
        or when it crosses into an already occupied cell in the StreamMask. The
        resulting trajectory is None if it is shorter than `minlength`.
        """

        stotal, x_traj, y_traj, z_traj = 0., [], [], []

        try:
            dmap.start_trajectory(x0, y0, z0)
        except InvalidIndexError:
            return None
        if integration_direction in ['both', 'backward']:
            s, xt, yt, zt = _integrate_rk12(x0, y0, z0, dmap, backward_time, maxlength)
            stotal += s
            x_traj += xt[::-1]
            y_traj += yt[::-1]
            z_traj += zt[::-1]

        if integration_direction in ['both', 'forward']:
            dmap.reset_start_point(x0, y0, z0)
            s, xt, yt, zt = _integrate_rk12(x0, y0, z0, dmap, forward_time, maxlength)
            if len(x_traj) > 0:
                xt = xt[1:]
                yt = yt[1:]
                zt = zt[1:]
            stotal += s
            x_traj += xt
            y_traj += yt
            z_traj += zt

        if stotal > minlength:
            return x_traj, y_traj, z_traj
        else:  # reject short trajectories
            dmap.undo_trajectory()
            return None

    return integrate


def _integrate_rk12(x0, y0, z0, dmap, f, maxlength):
    """2nd-order Runge-Kutta algorithm with adaptive step size.

    This method is also referred to as the improved Euler's method, or Heun's
    method. This method is favored over higher-order methods because:

    1. To get decent looking trajectories and to sample every mask cell
       on the trajectory we need a small timestep, so a lower order
       solver doesn't hurt us unless the data is *very* high resolution.
       In fact, for cases where the user inputs
       data smaller or of similar grid size to the mask grid, the higher
       order corrections are negligible because of the very fast linear
       interpolation used in `interpgrid`.

    2. For high resolution input data (i.e. beyond the mask
       resolution), we must reduce the timestep. Therefore, an adaptive
       timestep is more suited to the problem as this would be very hard
       to judge automatically otherwise.

    This integrator is about 1.5 - 2x as fast as both the RK4 and RK45
    solvers in most setups on my machine. I would recommend removing the
    other two to keep things simple.
    """
    # This error is below that needed to match the RK4 integrator. It
    # is set for visual reasons -- too low and corners start
    # appearing ugly and jagged. Can be tuned.
    maxerror = 0.003

    # This limit is important (for all integrators) to avoid the
    # trajectory skipping some mask cells. We could relax this
    # condition if we use the code which is commented out below to
    # increment the location gradually. However, due to the efficient
    # nature of the interpolation, this doesn't boost speed by much
    # for quite a bit of complexity.
    maxds = min(1. / dmap.mask.nx, 1. / dmap.mask.ny, 1. / dmap.mask.nz, 0.1)

    ds = maxds
    stotal = 0
    xi = x0
    yi = y0
    zi = z0
    xf_traj = []
    yf_traj = []
    zf_traj = []

    while dmap.grid.within_grid(xi, yi, zi):
        xf_traj.append(xi)
        yf_traj.append(yi)
        zf_traj.append(zi)
        try:
            k1x, k1y, k1z = f(xi, yi, zi)
            k2x, k2y, k2z = f(xi + ds * k1x,
                         yi + ds * k1y, zi + ds * k1z)
        except IndexError:
            # Out of the domain on one of the intermediate integration steps.
            # Take an Euler step to the boundary to improve neatness.
            ds, xf_traj, yf_traj, zf_traj = _euler_step(xf_traj, yf_traj, zf_traj, dmap, f)
            stotal += ds
            break
        except TerminateTrajectory:
            break

        dx1 = ds * k1x
        dy1 = ds * k1y
        dz1 = ds * k1z
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)
        dz2 = ds * 0.5 * (k1z + k2z)

        nx, ny, nz = dmap.grid.shape
        # Error is normalized to the axes coordinates
        error = np.sqrt(((dx2 - dx1) / nx)**2 + ((dy2 - dy1) / ny)**2 + ((dz2 - dz1) / nz)**2)

        # Only save step if within error tolerance
        if error < maxerror:
            xi += dx2
            yi += dy2
            zi += dz2
            try:
                dmap.update_trajectory(xi, yi, zi)
            except InvalidIndexError:
                break
            if stotal + ds > maxlength:
                break
            stotal += ds

        # recalculate stepsize based on step error
        if error == 0:
            ds = maxds
        else:
            ds = min(maxds, 0.85 * ds * (maxerror / error) ** 0.5)

    return stotal, xf_traj, yf_traj, zf_traj


def _euler_step(xf_traj, yf_traj, zf_traj, dmap, f):
    """Simple Euler integration step that extends streamline to boundary."""
    ny, nx, nz = dmap.grid.shape
    xi = xf_traj[-1]
    yi = yf_traj[-1]
    zi = zf_traj[-1]
    cx, cy, cz = f(xi, yi, zi)
    if cx == 0:
        dsx = np.inf
    elif cx < 0:
        dsx = xi / -cx
    else:
        dsx = (nx - 1 - xi) / cx
    if cy == 0:
        dsy = np.inf
    elif cy < 0:
        dsy = yi / -cy
    else:
        dsy = (ny - 1 - yi) / cy
    if cz == 0:
        dsz = np.inf
    elif cz < 0:
        dsz = zi / -cz
    else:
        dsz = (nz - 1 - zi) / cz
    ds = min(dsx, dsy, dsz)
    xf_traj.append(xi + cx * ds)
    yf_traj.append(yi + cy * ds)
    zf_traj.append(zi + cz * ds)
    return ds, xf_traj, yf_traj, zf_traj


# Utility functions
# ========================

def interpgrid(a, xi, yi, zi):
    """Fast 3D, linear interpolation on an integer grid"""

    Ny, Nx, Nz = np.shape(a)
    if isinstance(xi, np.ndarray):
        x = xi.astype(int)
        y = yi.astype(int)
        z = zi.astype(int)
        # Check that xn, yn don't exceed max index
        xn = np.clip(x + 1, 0, Nx - 1)
        yn = np.clip(y + 1, 0, Ny - 1)
        zn = np.clip(z + 1, 0, Nz - 1)
    else:
        x = int(xi)
        y = int(yi)
        z = int(zi)
        # conditional is faster than clipping for integers
        if x == (Nx - 1):
            xn = x
        else:
            xn = x + 1
        if y == (Ny - 1):
            yn = y
        else:
            yn = y + 1
        if z == (Nz - 1):
            zn = z
        else:
            zn = z + 1

    c000 = a[y, x, z]
    c100 = a[y, xn, z]
    c001 = a[y, x, zn]
    c101 = a[y, xn, zn]
    c010 = a[yn, x, z]
    c110 = a[xn, yn, z]
    c011 = a[yn, x, zn]
    c111 = a[yn, xn, zn]

    xt = xi - x
    yt = yi - y
    zt = zi - z

    c00 = c000 * (1 - xt) + c100 * xt
    c01 = c000 * (1 - xt) + c101 * xt
    c10 = c010 * (1 - xt) + c110 * xt
    c11 = c011 * (1 - xt) + c111 * xt

    c0 = c00 * (1 - yt) + c10 * yt
    c1 = c01 * (1 - yt) + c11 * yt
    c  = c0 * (1 - zt) + c1 * zt

    if not isinstance(xi, np.ndarray):
        if np.ma.is_masked(c):
            raise TerminateTrajectory

    return c


def _gen_starting_points(shape):
    """Yield starting points for streamlines.

    Trying points on the boundary first gives higher quality streamlines.
    This algorithm starts with a point on the mask corner and spirals inward.
    This algorithm is inefficient, but fast compared to rest of streamplot.
    """
    ny, nx, nz= shape
    xfirst = 0
    yfirst = 1
    zfirst = 1
    xlast = nx - 1
    ylast = ny - 1
    zlast = nz - 1
    x, y, z = 0, 0, 0
    verticaldirection = 1
    direction = 'right'
    for i in range(nx * ny * nz):

        yield x, y, z

        if direction == 'right':
            x += 1
            if x >= xlast:
                direction = 'up'
        elif direction == 'up':
            y += 1
            if y >= ylast:
                direction = 'left'
        elif direction == 'left':
            x -= 1
            if x <= xfirst:
                direction = 'down'
        elif direction == 'down':
            y -= 1
            if y <= yfirst:
                direction = 'vertical'
        elif direction == 'vertical':
            direction = 'right'
            if z >= zlast or ( z == 0 and verticaldirection < 0):
                xlast -= 1
                ylast -= 1
                xfirst += 1
                yfirst += 1
                verticaldirection = -1 * verticaldirection
            z += verticaldirection
