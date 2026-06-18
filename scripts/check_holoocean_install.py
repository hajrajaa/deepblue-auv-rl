import sys 


def main():

    try:
        import holoocean 
        from holoocean import packagemanager 
    
    except Exception as e:
        print("Failed to import HoloOcean.")
        print("Error:", repr(e))
        sys.exit(1)

    print("HoloOcean imported successfully.")
    print("HoloOcean module path:" ,getattr(holoocean, "__file__", "unknown"))


    try:
        print("\nInstalled packages:")
        print(packagemanager.installed_packages())

    except Exception as e:
        print("Could not read installed packages:", repr(e))

    
    try:
        print("\nAvailable packages:")
        print(packagemanager.available_packages())
    except Exception as e:
        print("Could not read available packages:", repr(e))

    
    try:
        print("\nOcean package info:")
        print(packagemanager.package_info("Ocean"))

    except Exception as e:
        print("Could not read Ocean package info:", repr(e))


if __name__ == "__main__":
    main()
    


    


