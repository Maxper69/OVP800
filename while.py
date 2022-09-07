import keyboard  # using module keyboard
while True:  # making a loop
    print("loop")
    try:  # used try so that if user pressed other than the given key error will not be shown
        print("try")
        if keyboard.is_pressed('q'):  # if key 'q' is pressed 
            print('You Pressed A Key!')
            break  # finishing the loop
    except:
        (print("except"))
        #break  # if user pressed a key other than the given key tpippip install keyboard
    