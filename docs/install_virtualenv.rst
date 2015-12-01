Install checklist on a new ubuntu/linux system with virtualenv
==============================================================

The checklist below should suffice to make OGGM work on any Debian based
system.


1. Linux packages
-----------------

For building stuffs::

    $ sudo apt-get install build-essential python-pip liblapack-dev gfortran libproj-dev

For matplolib to work on python2::

    $ sudo apt-get install python-gtk2-dev

And on python3::

    $ sudo apt-get install tk-dev python3-tk python3-dev

For GDAL::

    $ sudo apt-get install gdal-bin libgdal-dev python-gdal

For NETCDF::

    $ sudo apt-get install netcdf-bin ncview python-netcdf


2. Virtual environment
----------------------

Install::

    $ sudo pip install virtualenvwrapper

Check that you have it done::

    $ which virtualenvwrapper.sh
    /usr/local/bin/virtualenvwrapper.sh

Create the directory where the virtual environments will be created::

    $ mkdir ~/.pyvirtualenvs

Add these three lines to the files: ~/.profile and ~/.bashrc::

    # Virtual environment options
    export WORKON_HOME=$HOME/.pyvirtualenvs
    source /usr/local/bin/virtualenvwrapper_lazy.sh

Reset your profile::

    $ . ~/.profile

Make the virtual environment with python2::

    $ mkvirtualenv py2 -p /usr/bin/python

Or python3::

    $ mkvirtualenv py3 -p /usr/bin/python3

Details: http://simononsoftware.com/virtualenv-tutorial-part-2/


3. Python Packages
------------------

Be sure to be on the working environment::

    $ workon py2

Install one by one the easy stuff::

   $ pip install numpy scipy pandas shapely

For Matplotlib and **Python2** we need to loink the libs in the virtual env::

    $ ln -sf /usr/lib/python2.7/dist-packages/{glib,gobject,cairo,gtk-2.0,pygtk.py,pygtk.pth} $VIRTUAL_ENV/lib/python2.7/site-packages
    $ pip install matplotlib

Details: http://www.stevenmaude.co.uk/2013/09/installing-matplotlib-in-virtualenv.html

For Matplotlib and **Python3** it doesnt seem to be necessary::

    $ pip install matplotlib

Check if plotting works by running these three lines in python::

    >>> import matplotlib.pyplot as plt
    >>> plt.plot([1,2,3])
    >>> plt.show()

If nothing shows-up, something got wrong.

For **GDAL**, it's also not straight forward. First, check which version of
GDAL is installed::

    $ dpkg -s libgdal-dev

The version (10, 11, ...) should match that of the python package. Install
using the system binaries::

    $ pip install gdal==1.10.0 --install-option="build_ext" --install-option="--include-dirs=/usr/include/gdal"
    $ pip install fiona --install-option="build_ext" --install-option="--include-dirs=/usr/include/gdal"

Details: http://tylerickson.blogspot.co.at/2011/09/installing-gdal-in-python-virtual.html

Install further stuffs::

    $ pip install pyproj Pillow geopandas netcdf4 scikit-image configobj joblib

And the external libraries::

    $ pip install git+https://github.com/fmaussion/motionless.git
    $ pip install git+https://github.com/fmaussion/salem.git
    $ pip install git+https://github.com/fmaussion/cleo.git


4. git and GitHub
--------------------

We need git::

    $ sudo apt-get install git
    $ git config --global user.name "John Doe"
    $ git config --global user.email johndoe@example.com

And we need to get a SSH key for not having to retype a password all the time.
Details: https://help.github.com/articles/generating-ssh-keys/

Once you added the key to GitHub, clone the repository where you want::

    $ git clone https://github.com/OGGM/oggm


5. Testing
----------

It's easier with nose::

    $ pip install nose

And in oggm's root directory::

    $ nosetests

6. PyCharm
----------

We like to use PyCharm: http://www.jetbrains.com/pycharm/

Set-up pycharm project properties to use the virtualenv. You should be done!
