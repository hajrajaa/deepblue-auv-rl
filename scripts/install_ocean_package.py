import sys 


def main():

    try:

        import holoocean 
        from holoocean import packagemanager

    except Exception as e:
        print("Failed to import HoloOcean.")
        print("Error:", repr(e))
        sys.exit(1)

    print("HoloOcean module path:" ,getattr(holoocean, "__file__", "unknown"))

    try:
        installed=packagemanager.installed_packages()
        print("Installed packages:", installed)

    except Exception as e:
        print("Could not list installed packages:", repr(e))
        installed = []

    try:
        available=packagemanager.available_packages()
        print("Available packages:", available)
    except Exception as e:
        print("Could not list available packages:", repr(e))
        

    if "Ocean" not in installed:
        print("\nInstalling Ocean package... This can take time, please be patient.")
        holoocean.install("Ocean")

    else:
        print("\nOcean package is already installed.")


    try:
        installed=packagemanager.installed_packages()
        print("Installed packages after installation attempt:", installed)
    except Exception as e:
        print("Could not verifiy installed packages:",repr(e))


if __name__ == "__main__":
    main()
    
        
