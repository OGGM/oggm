import os
import sys
import getpass
import requests


def main():
    print("Enter your credentials for https://sso.eoc.dlr.de/pwm-tdmdem90")
    username = input("Username: ")
    password = getpass.getpass("Password: ")

    testurl = ("https://download.geoservice.dlr.de"
               "/TDM90/files/N51/E000/TDM1_DEM__30_N51E009.zip")
    r = requests.head(testurl, auth=(username, password))

    if r.status_code != 200:
        print("Authentication failed!")
        sys.exit(-1)

    print("Authentication successful!")

    tdmauthfile = os.path.expanduser("~/.tdmdem90.creds")
    with open(tdmauthfile, "w") as f:
        print(username, file=f)
        print(password, file=f)

    print("Credentials written to " + tdmauthfile)

    os.chmod(tdmauthfile, 0o600)

    sys.exit(0)
