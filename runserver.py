"""
@author:  zhaominyi heyun
@contact: 20210240011@fudan.edu.cn 20210240010@fudan.edu.cn
"""
from server import Server
import traceback
import os

if __name__ == '__main__':
    while True:
        server = Server()
        try:
            server.monitor_keyboard()
            server.setup()
            server.handle_rtsp_requests()
        except Exception as e:
            server.server_state = server.STATE.TEARDOWN
            print("Exception:",e)
            traceback.print_stack(e)
            os._exit(233)