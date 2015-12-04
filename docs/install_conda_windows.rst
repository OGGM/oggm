Install checklist on a Windows system with Anaconda
===================================================

**NOTE**: as of 04.12.2015 the method below is known to work well but the
standard method (`OGGM Install checklist with conda <./install_conda.rst>`_)
is the one we use for testing and is a bit easier.


1. Install Anaconda
-------------------

Go to the `Anaconda Windows Download page <https://www.continuum.io/downloads>`_ an download the 32-Bit/64-Bit version of the Anaconda executable installer to any directory you wish.
Double-click on the installer and follow the instructions given.


2. git and GitHub
--------------------

We also need git. You can get it `here <https://git-scm.com/download/win>`_.

Again, click on the excutable and follow the install instructions. Be sure to check the option ``Use git from Windows Command Prompt``.


3. Virtual environment
----------------------

Open an Anaconda Command Prompt (e.g. type 'Anaconda' in the search window of Windows 10) and browse to a comfortable working directory. To create a new virtual environment called "oggm", type::
    
    > conda create --name oggm

and ``y`` to continue the creation as soon as Anaconda asks you for it.


4. Python Packages
------------------


Be sure to be on the working environment that you have created::

    > activate oggm

Now you can install the Python packages you need. Some work straightforward::

    > conda install numpy scipy pandas matplotlib Pillow joblib netCDF4
    > conda install rasterio scikit-image configobj nose

Some are a bit tricky, and neither  ``conda install`` nor ``pip install`` works (or it *may* install correctly but the tools do not work properly). You should fall back to Christoph Gohlke's wheel files:

http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely : Choose version 1.5.13

http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyproj : Choose version 1.9.4

http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal : Choose version 1.11.3

http://www.lfd.uci.edu/~gohlke/pythonlibs/#fiona : Choose version 1.6.2

Save them to your working directory and install them then with pip by typing::

    > pip install wheel <filename.whl>

Only after the successful installation of these packages, do the last step and install ``geopandas``::

    > pip install geopandas



5. Install fmaussion's packages
--------------------------------

Make use of your frehsly installed git version and type::

    > pip install git+https://github.com/fmaussion/motionless.git
    > pip install git+https://github.com/fmaussion/salem.git
    > pip install git+https://github.com/fmaussion/cleo.git  

You have now installed all needed Python packages :-) 


6. Simple tests
---------------

To see whether or not the tricky part with the wheels has worked, type in the Anaconda Command Prompt::

    > activate oggm
    > python
    >>> import pyproj
    >>> from osgeo import gdal
    >>> import fiona
    >>> import shapely

If no error appears, everything has worked fine.

A nice test is to plot an image of Innsbruck, Austria, using Google Maps::

    >>> g = salem.GoogleCenterMap(center_ll=(11.38, 47.26), zoom=9)
    >>> m = cleo.Map(g.grid)
    >>> m.set_lonlat_countours(interval=100)
    >>> m.set_rgb(g.get_vardata())
    >>> ax = plt.axes()
    >>> m.plot(ax)
    >>> plt.show()
    >>> quit()

7. Testing OGGM
---------------

To test if all features of OGGM are working properly, use ``nose``::

    > conda install nose

and be sure to be in the oggm root directory. Then type::

    > nosetests

This will run all tests in the oggm directory. The tests might print out some warnings (most of them unrelated to OGGM) but all tests should also pass on windows.
