from serial import Serial

def main():
    print('Port opening... ')
    port = Serial('COM10', timeout=2)
    print('Port opened. ')
    with open('log.csv', 'w') as f:
        while True:
            line = port.readline().strip().decode()
            print('', line)
            print(line, file=f)

main()
