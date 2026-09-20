import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from perception.responsive_ui import blank_projector, run_check


class ResponsiveTests(unittest.TestCase):
    def test_blanking_does_not_swallow_keyboard_input(self):
        with patch('perception.responsive_ui.cv2.imshow') as show, \
             patch('perception.responsive_ui.cv2.waitKey') as keys:
            blank_projector(SimpleNamespace(name='Projector output'))
        show.assert_called_once()
        keys.assert_not_called()

    def test_slow_check_pumps_events_on_main_thread(self):
        event=threading.Event()
        main=threading.get_ident()
        def calculate():
            self.assertNotEqual(threading.get_ident(),main)
            self.assertTrue(event.wait(2))
            return 'PLACEMENT OK'
        def key(delay):
            self.assertEqual(threading.get_ident(),main)
            event.set()
            return -1
        with patch('perception.responsive_ui.show_preview'), \
             patch('perception.responsive_ui.cv2.waitKey',side_effect=key) as keys:
            self.assertEqual(run_check(np.zeros((10,10,3),np.uint8),'Checking',calculate),'PLACEMENT OK')
        self.assertGreater(keys.call_count,0)

    def test_worker_failure_reaches_normal_error_handler(self):
        def failure():
            raise ValueError('No new blue part')
        with patch('perception.responsive_ui.show_preview'), \
             patch('perception.responsive_ui.cv2.waitKey',return_value=-1):
            with self.assertRaisesRegex(ValueError,'No new blue'):
                run_check(np.zeros((10,10,3),np.uint8),'Checking',failure)
