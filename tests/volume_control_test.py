from __future__ import print_function

import datetime
from time import sleep

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL


def main():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

    volume = cast(interface, POINTER(IAudioEndpointVolume))
    #print(volume.GetMute())
    print("Before", volume.GetMasterVolumeLevel())
    #range=volume.GetVolumeRange()
    #print(range)
    #volume.SetMasterVolumeLevel(range[1], None)
    #print("After", volume.GetMasterVolumeLevel())


if __name__ == "__main__":
    main()
