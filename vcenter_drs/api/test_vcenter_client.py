from vcenter_client_pyvomi import VCenterPyVmomiClient

if __name__ == "__main__":
    client = VCenterPyVmomiClient()
    try:
        si = client.connect()
        print("Successfully connected to vCenter.")
        print("vCenter server time:", si.CurrentTime())
    except Exception as e:
        print(f"Failed to connect to vCenter: {e}")
    finally:
        client.disconnect() 