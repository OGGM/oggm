# Read in the RGI file(s)
import pandas as pd

rgi_dir = '/home/mowglie/disk/Data/GIS/SHAPES/RGI/RGI_V5/'
itmix_data_dir = '/home/mowglie/disk/Data/ITMIX/glaciers_sorted/'
df_itmix = pd.read_pickle('/home/mowglie/disk/Dropbox/Photos/itmix_rgi_plots/links/itmix_rgi_links.pkl')