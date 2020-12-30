
class Config:
    # The fps of the video
    DEFAULT_FPS = 30
    # The session id
    SESSION_ID = '123456'
    # The host of the server
    HOST = '127.0.0.1' # '10.30.5.78'

    LOCAL_HOST = '127.0.0.1' # '10.30.5.78'
    # The size of each chunk
    CHUNK_SIZE = 4096
    # Limit of the time
    TIMEOUT = 100
    # Port of the server
    SERVER_PORT = 23333
    # Delay time of the server
    RECV_DELAY = 20

    PACKET_HEADER_LENGTH = 5

    DEFAULT_FILE = './1.mp4'