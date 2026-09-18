import grpc
from typing import Optional

import threading
from collections import deque
from sys import platform
import socket
import time
from concurrent.futures import ThreadPoolExecutor

from .communicator import Communicator, PollCallback
from mlagents_envs.communicator_objects.unity_to_external_pb2_grpc import (
    UnityToExternalProtoServicer,
    add_UnityToExternalProtoServicer_to_server,
)
from mlagents_envs.communicator_objects.unity_message_pb2 import UnityMessageProto
from mlagents_envs.communicator_objects.unity_input_pb2 import UnityInputProto
from mlagents_envs.communicator_objects.unity_output_pb2 import UnityOutputProto
from .exception import UnityTimeOutException, UnityWorkerInUseException


class _ThreadHandoffChannel:
    """
    A single-direction, single-item-at-a-time handoff between two threads in
    the same process, e.g. a single-slot queue with a non-destructive poll.
    """

    def __init__(self):
        self._items: "deque" = deque()
        self._cond = threading.Condition()

    def put(self, item) -> None:
        with self._cond:
            self._items.append(item)
            self._cond.notify()

    def get(self):
        with self._cond:
            while not self._items:
                self._cond.wait()
            return self._items.popleft()

    def poll(self, timeout: Optional[float] = None) -> bool:
        """Non-destructive check for whether an item is waiting."""
        with self._cond:
            if self._items:
                return True
            self._cond.wait(timeout)
            return bool(self._items)


class _QueueConn:
    """
    One directional endpoint of a same-process, cross-thread handoff between
    the gRPC worker thread and the main training thread. Stands in for one
    end of a multiprocessing.Pipe, which pickles every message and makes a
    real OS pipe syscall - overhead that only makes sense for actual
    cross-process IPC. Both threads here share one interpreter, so an
    in-memory, GIL-protected handoff does the same job without the pickling
    or the syscalls.
    """

    def __init__(self, send_ch: _ThreadHandoffChannel, recv_ch: _ThreadHandoffChannel):
        self._send_ch = send_ch
        self._recv_ch = recv_ch

    def send(self, obj) -> None:
        self._send_ch.put(obj)

    def recv(self):
        return self._recv_ch.get()

    def poll(self, timeout: Optional[float] = None) -> bool:
        return self._recv_ch.poll(timeout)

    def close(self) -> None:
        # No OS resource to release - this is an in-memory, same-process
        # handoff, unlike the multiprocessing.Pipe connection it replaces.
        pass


class UnityToExternalServicerImplementation(UnityToExternalProtoServicer):
    def __init__(self):
        to_parent = _ThreadHandoffChannel()
        to_child = _ThreadHandoffChannel()
        self.parent_conn = _QueueConn(send_ch=to_child, recv_ch=to_parent)
        self.child_conn = _QueueConn(send_ch=to_parent, recv_ch=to_child)

    def Initialize(self, request, context):
        self.child_conn.send(request)
        return self.child_conn.recv()

    def Exchange(self, request, context):
        self.child_conn.send(request)
        return self.child_conn.recv()


class RpcCommunicator(Communicator):
    def __init__(self, worker_id=0, base_port=5005, timeout_wait=30):
        """
        Python side of the grpc communication. Python is the server and Unity the client


        :int base_port: Baseline port number to connect to Unity environment over. worker_id increments over this.
        :int worker_id: Offset from base_port. Used for training multiple environments simultaneously.
        :int timeout_wait: Timeout (in seconds) to wait for a response before exiting.
        """
        super().__init__(worker_id, base_port)
        self.port = base_port + worker_id
        self.worker_id = worker_id
        self.timeout_wait = timeout_wait
        self.server = None
        self.unity_to_external = None
        self.is_open = False
        self.create_server()

    def create_server(self):
        """
        Creates the GRPC server.
        """
        self.check_port(self.port)

        try:
            # Establish communication grpc
            self.server = grpc.server(
                thread_pool=ThreadPoolExecutor(max_workers=10),
                options=(("grpc.so_reuseport", 1),),
            )
            self.unity_to_external = UnityToExternalServicerImplementation()
            add_UnityToExternalProtoServicer_to_server(
                self.unity_to_external, self.server
            )
            # Using unspecified address, which means that grpc is communicating on all IPs
            # This is so that the docker container can connect.
            self.server.add_insecure_port("[::]:" + str(self.port))
            self.server.start()
            self.is_open = True
        except Exception:
            raise UnityWorkerInUseException(self.worker_id)

    def check_port(self, port):
        """
        Attempts to bind to the requested communicator port, checking if it is already in use.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if platform == "linux" or platform == "linux2":
            # On linux, the port remains unusable for TIME_WAIT=60 seconds after closing
            # SO_REUSEADDR frees the port right after closing the environment
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("localhost", port))
        except OSError:
            raise UnityWorkerInUseException(self.worker_id)
        finally:
            s.close()

    def poll_for_timeout(self, poll_callback: Optional[PollCallback] = None) -> None:
        """
        Polls the GRPC parent connection for data, to be used before calling recv.  This prevents
        us from hanging indefinitely in the case where the environment process has died or was not
        launched.

        Additionally, a callback can be passed to periodically check the state of the environment.
        This is used to detect the case when the environment dies without cleaning up the connection,
        so that we can stop sooner and raise a more appropriate error.
        """
        deadline = time.monotonic() + self.timeout_wait
        callback_timeout_wait = self.timeout_wait // 10
        while time.monotonic() < deadline:
            if self.unity_to_external.parent_conn.poll(callback_timeout_wait):
                # Got an acknowledgment from the connection
                return
            if poll_callback:
                # Fire the callback - if it detects something wrong, it should raise an exception.
                poll_callback()

        # Got this far without reading any data from the connection, so it must be dead.
        raise UnityTimeOutException(
            "The Unity environment took too long to respond. Make sure that :\n"
            "\t The environment does not need user interaction to launch\n"
            '\t The Agents\' Behavior Parameters > Behavior Type is set to "Default"\n'
            "\t The environment and the Python interface have compatible versions.\n"
            "\t If you're running on a headless server without graphics support, turn off display "
            "by either passing --no-graphics option or build your Unity executable as server build."
        )

    def initialize(
        self, inputs: UnityInputProto, poll_callback: Optional[PollCallback] = None
    ) -> UnityOutputProto:
        self.poll_for_timeout(poll_callback)
        aca_param = self.unity_to_external.parent_conn.recv().unity_output
        message = UnityMessageProto()
        message.header.status = 200
        message.unity_input.CopyFrom(inputs)
        self.unity_to_external.parent_conn.send(message)
        self.unity_to_external.parent_conn.recv()
        return aca_param

    def exchange(
        self, inputs: UnityInputProto, poll_callback: Optional[PollCallback] = None
    ) -> Optional[UnityOutputProto]:
        message = UnityMessageProto()
        message.header.status = 200
        message.unity_input.CopyFrom(inputs)
        self.unity_to_external.parent_conn.send(message)
        self.poll_for_timeout(poll_callback)
        output = self.unity_to_external.parent_conn.recv()
        if output.header.status != 200:
            return None
        return output.unity_output

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the grpc connection.
        """
        if self.is_open:
            message_input = UnityMessageProto()
            message_input.header.status = 400
            self.unity_to_external.parent_conn.send(message_input)
            self.unity_to_external.parent_conn.close()
            self.server.stop(False)
            self.is_open = False
