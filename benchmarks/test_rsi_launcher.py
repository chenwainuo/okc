import subprocess
import unittest
from run_rsi_h200 import container_absence_confirmed


class ContainerTerminationEvidenceTest(unittest.TestCase):
    def test_docker_absence_case_variants(self):
        for message in ('Error: No such object: test', 'error: no such object: test',
                        'Error response from daemon: No such container: test'):
            with self.subTest(message=message):
                self.assertTrue(container_absence_confirmed(
                    subprocess.CompletedProcess([], 1, '', message)))

    def test_daemon_failure_or_existing_container_is_not_absence(self):
        for code, message in ((1, 'Cannot connect to the Docker daemon'),
                              (1, 'permission denied'), (0, 'No such object: test'), (1, '')):
            with self.subTest(code=code, message=message):
                self.assertFalse(container_absence_confirmed(
                    subprocess.CompletedProcess([], code, '', message)))
