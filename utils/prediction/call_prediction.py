import pyttsx3

limb_names = {0: 'left foot',
              1: 'right hand',
              2: 'left foot',
              3: 'right foot'}


class TextToSpeech:
    def __init__(self, rate=150, voice_index=1):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[voice_index].id)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def stop(self):
        self.engine.stop()


def format_command(limb, direction, distance):
    """
    Generate a speech string based on input parameters.
    :param limb: 0-3 corresponding to which limb should move.
    :param direction: The clock position (e.g., 12, 3, etc.).
    :param distance: The distance in inches.
    :return: A formatted string to be spoken.
    """
    # calculate feet and inches
    feet = distance // 12
    inches = distance % 12
    foot_word = "foot" if feet == 1 else "feet"

    if feet == 0 and inches == 1:
        return f"move {limb_names[limb]} {direction} o'clock {inches} inch"
    elif feet == 0:
        return f"move {limb_names[limb]} {direction} o'clock {inches} inches"
    elif inches < 4:
        return f"move {limb_names[limb]} {direction} o'clock about {feet} {foot_word}"
    elif inches < 10:
        return f"move {limb_names[limb]} {direction} o'clock {feet} and a half feet"
    else:
        return f"move {limb_names[limb]} {direction} o'clock about {feet + 1} feet"


def call_prediction(tts, limb, direction, distance):
    """
        Call out the predicted movement.
        :param limb: 0-3 corresponding to which limb should move.
        :param direction: The clock position (e.g., 12, 3, etc.).
        :param distance: The distance in inches.
        """
    command = format_command(limb, direction, distance)
    print(f'\tCommand: {command}')
    tts.speak(command)


if __name__ == '__main__':
    tts = TextToSpeech()
    distances = [0, 1, 2, 12, 17, 23, 25, 27]
    limb_labels = [0, 1, 2, 3]
    directions = [10, 11, 12, 1, 2, 3]

    limb = 1
    call_prediction(tts, limb, directions[-2], distances[-2])
    tts.stop()
