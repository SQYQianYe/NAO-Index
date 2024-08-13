import os
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from eofs.standard import Eof

Lisbon = [-9, 39]
Stykkisholmur = [-22.5, 65]
def plot_background(ax):
    ax.coastlines()
    ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='gray', alpha=0.5, linestyle='-')
    boundary_path = make_boundary_path(lon, lat)
    ax.set_boundary(boundary_path, transform=ccrs.PlateCarree())
    return ax

def make_boundary_path(lon,lat):
    lons,lats=np.meshgrid(lon,lat)
    boundary_path = np.array([lons[-1,:],lats[-1,:]])
    boundary_path = np.append(boundary_path,np.array([lons[::-1,-1],lats[::-1,-1]]),axis=1)
    boundary_path = np.append(boundary_path,np.array([lons[1,::-1],lats[1,::-1]]),axis=1)
    boundary_path = np.append(boundary_path,np.array([lons[:,1],lats[:,1]]),axis=1)
    boundary_path = mpath.Path(np.swapaxes(boundary_path, 0, 1))
    return boundary_path

def plot_town(ax):
    ax.plot(Lisbon[0], Lisbon[1], marker='o', color='red', markersize=8,
            alpha=0.7, transform=ccrs.PlateCarree())
    ax.text(Lisbon[0], Lisbon[1], u'Lisbon',
            verticalalignment='center', horizontalalignment='right',
            transform=ccrs.PlateCarree()._as_mpl_transform(ax),
            bbox=dict(facecolor='sandybrown', alpha=0.5, boxstyle='round'))
    ax.plot(Stykkisholmur[0], Stykkisholmur[1], marker='o', color='red', markersize=8,
            alpha=0.7, transform=ccrs.PlateCarree())
    ax.text(Stykkisholmur[0], Stykkisholmur[1], u'Stykkisholmur',
            verticalalignment='center', horizontalalignment='right',
            transform=ccrs.PlateCarree()._as_mpl_transform(ax),
            bbox=dict(facecolor='sandybrown', alpha=0.5, boxstyle='round'))

def plot_pc(ax):
    plt.xlabel('Year')
    plt.axhline(0, color='k')
    plt.xlim(int(yr1), int(yr2)+1)
    plt.xticks(xi, range(int(yr1), int(yr2)+1,5))
    ax.axvline(1950, color='grey', linestyle='--', linewidth=0.5)
    ax.axvline(1960, color='grey', linestyle='--', linewidth=0.5)
    ax.axvline(1970, color='grey', linestyle='--', linewidth=0.5)
    ax.axvline(1980, color='grey', linestyle='--', linewidth=0.5)
    ax.axvline(1990, color='grey', linestyle='--', linewidth=0.5)
    ax.axvline(2000, color='grey', linestyle='--', linewidth=0.5)
    ax.axvline(2010, color='grey', linestyle='--', linewidth=0.5)
    plt.ylim(-1500, 1500)
    return ax
dir_data='./data/'
dir_figs='./figs/'
if not os.path.exists(dir_figs):
    os.makedirs(dir_figs)

nc_file = dir_data+'era5.mslp.djfm.mean.natl.nc'
data    = xr.open_dataset(nc_file)
latS=20
latN=80
lonW=-80
lonE=30
yr1 = '1981'
yr2 = '2020'
years = np.arange(int(yr1), int(yr2)+1)
xi = [i for i in range(int(yr1), int(yr2)+1,5)]

data    = xr.open_dataset(nc_file).sel(latitude=slice(latN,latS)).sel(longitude=slice(lonW,lonE)).sel(time=slice(yr1,yr2))
data_clim = data.mean('time')
print(data)
print(data_clim)

lat  = data.latitude.values
lon  = data.longitude.values
time  = data.time.values

slp, slp_clim, = xr.broadcast(data['msl']/100, data_clim['msl']/100)

#--  Manage dates
time_str=[x for x in range(len(time))]
date_str=[x for x in range(len(time))]
for i in range(len(time)):
	time_str[i] = str(time[i])
	date_str[i] = time_str[i][0:10]

slp_anom=(slp-slp_clim)

levels = np.arange(980,1042,2)
levels_a = np.arange(-16,18,2)

projection = ccrs.Orthographic(central_longitude=(lonW+lonE)/2, central_latitude=(latS+latN)/2)

fig = plt.figure(figsize=(15., 10.), dpi = 300)
ax = fig.add_subplot(1, 1, 1, projection=projection)
ax.set_title('Mean Sea Level Pressure (hPa) - DJFM '+ yr1+'-'+yr2, loc='center')
plot_background(ax)
cf = ax.contourf(lon, lat, slp_clim[0,:,:], levels, cmap='jet', transform=ccrs.PlateCarree(), extend='both')
c = ax.contour(lon, lat, slp_clim[0,:,:], levels, colors='black', linewidths=0.2, transform=ccrs.PlateCarree())
cb = fig.colorbar(cf, orientation='horizontal', aspect=65, shrink=0.5, pad=0.05, extendrect='True')
cb.set_label('hPa', size='large')
plot_town(ax)

plt.show()



wgts = np.sqrt(np.cos(np.deg2rad(lat)))[:, np.newaxis]
solver = Eof(np.array(slp_anom), weights=wgts, center=True)
eigenvalues = solver.eigenvalues()
total_variance = solver.totalAnomalyVariance()
varfrac = solver.varianceFraction()
eofs = solver.eofs()
pcs = solver.pcs()
pcs_norm = solver.pcs(pcscaling=1)
clevs = np.linspace(-0.005, 0.005, 21)


fig = plt.figure(figsize=(20, 8), dpi = 300)
fig.suptitle('EOF1 and PC1 : MSLP DJFM '+yr1+'-'+yr2, fontsize=16)
ax = fig.add_subplot(121, projection=projection)
plt.title('EOF1 ('+str(int(varfrac[0]*100))+'%)', fontsize=10, loc='center')
plot_background(ax)
plot_town(ax)
cf = ax.contourf(lon, lat, eofs[0], levels=clevs, transform=ccrs.PlateCarree(), cmap='RdBu_r', extend='both')
c = ax.contour(lon, lat, eofs[0], levels=clevs, colors='black', linewidths=1, transform=ccrs.PlateCarree())
cb = fig.colorbar(cf, orientation='horizontal', aspect=65, shrink=1, pad=0.05, extendrect='True')
ax = fig.add_subplot(122)
plt.ylabel('PC1')
plot_pc(ax)
colormat=np.where(pcs[:,0]>0, 'red','blue')
plt.bar(years, pcs[:,0], width=1, color=colormat, edgecolor = 'k')
plt.show()


fig = plt.figure(figsize=(15., 8.), dpi = 300)
nao_file=dir_data+'nao_station_djfm_0.txt'

X, Y = [], []
for line in open(nao_file, 'r'):
    values = [float(s) for s in line.split()]
    X.append(values[0])
    Y.append(values[1])
pc1=pcs_norm[:,0]
XX=np.asarray(X)
YY=np.asarray(Y)
id=np.where((XX >= int(yr1)) & (XX <= int(yr2)))
nao_year=XX[id]
nao_index=YY[id]
cor=np.corrcoef(pc1, nao_index)
plt.title('Normalized PC1 VS NAO station-based index : '+yr1+'-'+yr2, loc='center')
plt.title('Correlation : '+str(round(cor[0,1],2)), loc='right')

plt.xlabel('Year')
plt.ylabel('Normalized Units')
yrs = range(int(yr1), int(yr2)+1)
xi = [i for i in range(int(yr1), int(yr2)+1,2)]
plt.plot(yrs, pc1, color='b', linewidth=2, label='PC1 from EOF analysis')
plt.plot(yrs, nao_index, color='r', linewidth=2, label='NAO index')
plt.axhline(0, color='k')
plt.axhline(2, color='grey', linestyle='--', linewidth=0.5)
plt.axhline(-2, color='grey', linestyle='--', linewidth=0.5)
plt.xlim(int(yr1), int(yr2)+1)
plt.xticks(xi, range(int(yr1), int(yr2)+1,2))
plt.ylim(-6, 6)
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.legend()

plt.show()
