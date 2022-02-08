import os
import sys
import getpass
import requests
from netrc import netrc
from subprocess import Popen
from urllib.parse import urlparse
import ftplib
import socket


def _test_credentials(authfile, key, testurl):
    """ Helper function to test the credentials
    """

    if 'ftps://' in testurl:
        # FTP_TLS needs some extra workflow
        try:
            upar = urlparse(testurl)

            # Decide if Implicit or Explicit FTPS
            if upar.port == 990:
                from oggm.utils import ImplicitFTPTLS
                ftps = ImplicitFTPTLS()
            elif upar.port == 21:
                ftps = ftplib.FTP_TLS()

            # establish ssl connection
            ftps.connect(host=upar.hostname, port=upar.port, timeout=30)
            ftps.login(user=netrc(authfile).authenticators(key)[0],
                       passwd=netrc(authfile).authenticators(key)[2])
            ftps.prot_p()
            ftps.close()
            print("Authentication successful!")
            return 0
        except (ftplib.error_perm, socket.timeout, socket.gaierror) as err:
            print("Authentication for {} failed with: {1}".
                  format(key, err))
            ftps.close()
            return -1

    else:
        # checking requests.head does work for some servers even with wrong
        # credentials -> use requests.get but only for some bytes
        header = {"Range": "bytes=0-100"}

        r = requests.get(testurl, headers=header,
                         auth=(netrc(authfile).authenticators(key)[0],
                               netrc(authfile).authenticators(key)[2]))
        if (r.status_code == 206) or (r.status_code == 200):
            # code 200 is "OK", code 206 is "partial content" which is ok here
            print("Authentication successful!")
            return 0
        else:
            print("Authentication failed with HTML status code {}!".
                  format(r.status_code))
            return -1


def read_credentials(key, testurl):
    """ Generic function to read credentials and attach them to a .netrc file

    Parameters
    ----------
    key : str
        credentials identifier
    testurl : str
        a filelink to test the credentials against
    """
    # the .netrc file
    authfile = os.path.expanduser("~/.netrc")

    # check if file and credentials already exist
    try:
        netrc(authfile).authenticators(key)[0]
        print("Credentials for {} already exist".format(key))
        test = _test_credentials(authfile, key, testurl)
        if test == 0:
            sys.exit(0)
        else:
            print("Existing credentials for {} do not work!".format(key))
            raise ValueError

    except (FileNotFoundError, TypeError, ValueError) as err:
        # no credentials, so we read them from the user
        print('Enter your credentials for {0} (this might override existing '
              'credentials for {0}!)'.format(key))
        username = input("Username: ")
        password = getpass.getpass("Password: ")

        homedir = os.path.expanduser("~")

        # if .netrc does not exist we create it first and set permissions
        if isinstance(err, FileNotFoundError):
            Popen('touch {0}.netrc | chmod 0600 {0}.netrc'.
                  format(homedir + os.sep), shell=True)
            print('Created credential file: {}.netrc'.format(homedir + os.sep))

        # if the existing credentials are wrong we delete them
        if isinstance(err, ValueError):
            with open(authfile, "r") as f:
                lines = f.readlines()
            iterlines = iter(lines)
            # rewrite the .netrc file without these credentials
            with open(authfile, "w") as f:
                for line in iterlines:
                    if line.strip() != "machine {}".format(key):
                        f.write(line)
                    else:
                        # skip the machine + username + password lines
                        next(iterlines)
                        next(iterlines)

        # we should have a .netrc file which is either empty or contains only
        # other credentials. Now add the new ones to the end of the file
        with open(authfile, 'a') as f:
            f.write('\nmachine {}\n'.format(key))
            f.write('login {}\n'.format(username))
            f.write('password {}\n'.format(password))

        print("Credentials for {} written to {}".format(key, authfile))

    sys.exit(_test_credentials(authfile, key, testurl))


def cli():
    """ command line interface to store different NETRC credentials
    call via 'oggm_netrc_credentials'
    """

    print('This will store login credentials in a local .netrc file.\n'
          'Enter the number or the service you want to add credentials for, '
          'this might override existing credentials in the .netrc file!\n\n'
          '[0] urs.earthdata.nasa.gov, ASTER DEM or NASADEM\n'
          '[1] geoservice.dlr.de, TanDEM-X\n'
          '[2] spacedata.copernicus.eu, Copernicus DEM Glo-90\n\n')
    nr = input("Number: ")

    if nr == '0':
        key = 'urs.earthdata.nasa.gov'
        testurl = ('https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/' +
                   '2000.03.01/ASTGTMV003_S09W158.zip')

    elif nr == '1':
        key = 'geoservice.dlr.de'
        testurl = ("https://download.geoservice.dlr.de" +
                   "/TDM90/files/N57/E000/TDM1_DEM__30_N57E006.zip")

    elif nr == '2':
        key = 'spacedata.copernicus.eu'
        testurl = 'ftps://cdsdata.copernicus.eu:990'

    else:
        print('Not a valid number, aborting.')
        sys.exit(-1)

    read_credentials(key, testurl)
