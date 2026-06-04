import time
import argparse
from smoking_analyzer import SmokingAnalyzer

parser = argparse.ArgumentParser(description='RTSP reconnect quick tester (does not load YOLO).')
parser.add_argument('--source', '-s', default='rtsp://127.0.0.1:554/invalid', help='RTSP source to test')
parser.add_argument('--attempts', '-a', type=int, default=3, help='Reconnect attempts')
parser.add_argument('--delay', '-d', type=int, default=2, help='Base delay seconds')
args = parser.parse_args()

print('Creating analyzer for source:', args.source)
ana = SmokingAnalyzer(source=args.source)

print('Opening capture directly...')
cap = ana._open_capture(args.source)
print('cap.isOpened():', cap.isOpened())

print('\nInvoking _reconnect_capture(max_attempts=%d, base_delay=%d)' % (args.attempts, args.delay))
ok = ana._reconnect_capture(max_attempts=args.attempts, base_delay=args.delay)
print('reconnect returned:', ok)
print('reconnect_attempts:', ana.reconnect_attempts)
print('last_reconnect_time:', ana.last_reconnect_time)
if ana.last_reconnect_time:
    print('last_reconnect_time (human):', time.ctime(ana.last_reconnect_time))

print('\nNow polling analyzer.status/_error_msg for 10s to observe state changes...')
for i in range(10):
    print('[%02d] status=%s error=%s' % (i, getattr(ana, '_status', None), getattr(ana, '_error_msg', None)))
    time.sleep(1)

print('\nTest complete')
