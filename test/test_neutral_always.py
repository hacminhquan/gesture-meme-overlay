import numpy as np

from main import GestureMemeApp


def test_neutral_shown_when_no_gesture():
    app = GestureMemeApp()
    # Ensure memes loaded (placeholders if images missing)
    app.overlay.load_memes()

    # Prevent actual hand detection (avoid dependency on camera/model)
    app.recognizer.detector.findHands = lambda frame: ([], frame)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = app.process_frame(frame)

    assert app.overlay.current_meme == "neutral"
    assert app.overlay.target_alpha == 1.0


def test_hand_swap_uses_second_hand_and_swaps_type():
    app = GestureMemeApp()
    app.overlay.load_memes()

    # Create two fake hands (dictionaries like cvzone provides)
    hand0 = {"type": "Left", "lmList": [[10, 200]] * 21, "bbox": [10, 150, 50, 100]}
    hand1 = {"type": "Right", "lmList": [[100, 200]] * 21, "bbox": [100, 150, 50, 100]}

    # Make fingersUp return no fingers (avoid triggering gestures)
    app.recognizer.detector.fingersUp = lambda h: [0, 0, 0, 0, 0]
    app.recognizer.detector.findHands = lambda frame: ([hand0, hand1], frame)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = app.process_frame(frame)

    # The recognizer should have swapped handedness on the selected (second) hand
    assert hand1["type"] == "Left"
    # Should still show neutral meme
    assert app.overlay.current_meme == "neutral"
