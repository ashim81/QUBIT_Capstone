"""Statevector and simulator checks for the notebook's public circuit builders.

Run: .venv/bin/python -m unittest discover -s tests -p test_quantum_library.py -v
Notebook definition cells are loaded without executing markdown examples or jobs.
"""
import itertools
import json
from pathlib import Path
import unittest

import nbformat
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector, partial_trace
from qiskit_aer import AerSimulator

NOTEBOOK = Path(__file__).resolve().parents[1] / "quantum_gate_algorithm_library.ipynb"
LIBRARY = {}
for cell in json.loads(NOTEBOOK.read_text())["cells"]:
    if cell["cell_type"] == "code":
        exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), LIBRARY)


def build(name, *args, **kwargs):
    return LIBRARY[name](*args, **kwargs)


class QuantumLibraryTests(unittest.TestCase):
    def assert_state(self, circuit, expected):
        np.testing.assert_allclose(Statevector.from_instruction(circuit).data, expected, atol=1e-10)

    def assert_basis(self, circuit, index):
        expected = np.zeros(2**circuit.num_qubits, dtype=complex)
        expected[index] = 1
        self.assertTrue(Statevector.from_instruction(circuit).equiv(Statevector(expected)))

    def test_notebook_and_default_contract(self):
        nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))
        builders = {name: function for name, function in LIBRARY.items()
                    if name.endswith('_circuit') and callable(function)}
        self.assertEqual(len(builders), 26)
        for name, function in builders.items():
            with self.subTest(name=name):
                first, second = function(), function()
                self.assertIsInstance(first, QuantumCircuit)
                self.assertIsNot(first, second)
                self.assertEqual(first.num_clbits, 0)
                self.assertFalse(any(item.operation.name in ('measure', 'reset') for item in first.data))
                self.assertAlmostEqual(np.sum(Statevector.from_instruction(first).probabilities()), 1)

    def test_original_gates_and_randomness(self):
        self.assert_basis(build('x_circuit'), 1)
        self.assert_state(build('y_circuit'), [0, 1j])
        self.assert_basis(build('z_circuit'), 0)
        self.assert_state(build('hadamard_circuit'), np.ones(2)/np.sqrt(2))
        self.assert_state(build('rx_circuit'), [0, -1j])
        for name, size in [('random_bit_circuit', 1), ('random_byte_circuit', 8)]:
            self.assert_state(build(name), np.ones(2**size)/np.sqrt(2**size))

    def test_deutsch_jozsa_and_bernstein_vazirani(self):
        for n in range(1, 5):
            for oracle in ('constant_0', 'constant_1', 'balanced'):
                probabilities = Statevector.from_instruction(build('deutsch_jozsa_circuit', n, oracle)).probabilities(list(range(n)))
                self.assertAlmostEqual(probabilities[0 if oracle.startswith('constant') else 2**n-1], 1)
            for bits in itertools.product('01', repeat=n):
                secret = ''.join(bits)
                probabilities = Statevector.from_instruction(build('bernstein_vazirani_circuit', secret)).probabilities(list(range(n)))
                self.assertAlmostEqual(probabilities[int(secret, 2)], 1)

    def test_sqrt_x_bell_and_ghz(self):
        np.testing.assert_allclose(Operator(build('sqrt_x_circuit', 2)).data, Operator(build('x_circuit')).data)
        np.testing.assert_allclose(Statevector.from_instruction(build('sqrt_x_circuit', 1)).probabilities(), [.5, .5])
        for name, expected in {
            'phi_plus': [1, 0, 0, 1], 'phi_minus': [1, 0, 0, -1],
            'psi_plus': [0, 1, 1, 0], 'psi_minus': [0, 1, -1, 0]
        }.items():
            self.assert_state(build('bell_state_circuit', name), np.array(expected)/np.sqrt(2))
        for n in range(2, 6):
            expected = np.zeros(2**n); expected[[0, -1]] = 1/np.sqrt(2)
            self.assert_state(build('ghz_circuit', n), expected)

    def test_phase_and_swap(self):
        for angle in (0, .3, np.pi, -np.pi/2):
            probabilities = Statevector.from_instruction(build('phase_kickback_circuit', angle)).probabilities([0])
            np.testing.assert_allclose(probabilities, [np.cos(angle/2)**2, np.sin(angle/2)**2], atol=1e-10)
            self.assert_state(build('conditional_phase_circuit', angle), np.array([1, 1, 1, np.exp(1j*angle)])/2)
        for a, b in ((0, 0), (0, np.pi), (.3, 1.7), (-.8, .2)):
            probabilities = Statevector.from_instruction(build('swap_test_circuit', a, b)).probabilities([0])
            self.assertAlmostEqual(probabilities[0], (1+np.cos((a-b)/2)**2)/2)

    def test_teleportation_preserves_relative_phase(self):
        for theta, phi in ((0, 0), (np.pi, 0), (np.pi/2, np.pi/2), (.71, -.89)):
            state = Statevector.from_instruction(build('teleportation_circuit', theta, phi))
            expected = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
            np.testing.assert_allclose(partial_trace(state, [0, 1]).data,
                                       np.outer(expected, expected.conj()), atol=1e-10)

    def test_increment_wraparound_and_inverse(self):
        for n in range(1, 4):
            N = 2**n
            for value in range(N):
                for amount in (-N-1, -1, 0, 1, 2, N+1):
                    self.assert_basis(build('increment_circuit', n, amount, value), (value+amount)%N)
            op = build('increment_circuit', n, 3, None)
            inverse = build('increment_circuit', n, -3, None)
            np.testing.assert_allclose(Operator(op.compose(inverse)).data, np.eye(N), atol=1e-10)

    def test_addition_and_squared_addition(self):
        for n in range(1, 4):
            N = 2**n
            for a, b in itertools.product(range(N), repeat=2):
                self.assert_basis(build('quantum_addition_circuit', n, a, b), b*N+(a+b)%N)
                self.assert_basis(build('add_squared_circuit', n, a, b), b*N+(a+b*b)%N)
            # Check the operators, including coherent phase preservation, on complex superpositions.
            rng = np.random.default_rng(42+n)
            amplitudes = rng.normal(size=N*N)+1j*rng.normal(size=N*N)
            amplitudes /= np.linalg.norm(amplitudes)
            for name, power in [('quantum_addition_circuit', 1), ('add_squared_circuit', 2)]:
                operation = build(name, n, 0, 0)
                expected = np.zeros(N*N, dtype=complex)
                for a, b in itertools.product(range(N), repeat=2):
                    expected[b*N+(a+b**power)%N] = amplitudes[b*N+a]
                np.testing.assert_allclose(Statevector(amplitudes).evolve(operation).data, expected, atol=1e-10)

    def test_conditional_arithmetic_and_xor(self):
        for n in range(1, 4):
            N = 2**n
            for initial, amount, flag in itertools.product(range(N), (-2, 0, 1), (0, 1)):
                self.assert_basis(build('conditional_increment_circuit', n, amount, initial, flag),
                                  2*((initial+flag*amount)%N)+flag)
            state = np.zeros(2*N, dtype=complex)
            state[0] = state[3] = 1/np.sqrt(2)
            self.assert_state(build('conditional_increment_circuit', n, 1, 0, None), state)
        for a, b in itertools.product((0, 1), repeat=2):
            self.assert_basis(build('reversible_xor_circuit', a, b), a+2*b+4*(a^b))

    def test_grover_analytic_probability(self):
        for n in range(1, 5):
            N = 2**n
            for count in sorted({1, N//2, N-1, N}):
                marked = [format(k, f'0{n}b') for k in range(count)]
                theta = np.arcsin(np.sqrt(count/N))
                for r in (0, 1, 2, None):
                    iterations = int(np.floor(np.pi/(4*theta))) if r is None else r
                    state = Statevector.from_instruction(build('grover_multi_circuit', marked, r))
                    expected = np.sin((2*iterations+1)*theta)**2
                    np.testing.assert_allclose(state.probabilities()[:count], expected/count, atol=1e-10)
            for k in range(N):
                circuit = build('grover_circuit', format(k, f'0{n}b'))
                theta = np.arcsin(1/np.sqrt(N)); r = int(np.floor(np.pi/(4*theta)))
                self.assertAlmostEqual(Statevector.from_instruction(circuit).probabilities()[k], np.sin((2*r+1)*theta)**2)

    def test_qft_signals_and_phase_estimation(self):
        for n in range(1, 5):
            N = 2**n
            fourier = np.exp(2j*np.pi*np.outer(np.arange(N), np.arange(N))/N)/np.sqrt(N)
            np.testing.assert_allclose(Operator(build('qft_circuit', n)).data, fourier, atol=1e-10)
            np.testing.assert_allclose(Operator(build('qft_circuit', n, True)).data, fourier.conj().T, atol=1e-10)
            for k in range(N):
                self.assert_basis(build('qft_signal_circuit', n, k), k)
                probabilities = Statevector.from_instruction(build('phase_estimation_circuit', k/N, n)).probabilities(list(range(n)))
                self.assertAlmostEqual(probabilities[k], 1)
            pulse = np.zeros(N); pulse[:N//2] = 1/np.sqrt(N//2)
            self.assert_state(build('qft_signal_circuit', n, 0, 'square'), fourier@pulse)
            phase = .3
            expected = [abs(np.mean(np.exp(2j*np.pi*np.arange(N)*(phase-y/N))))**2 for y in range(N)]
            actual = Statevector.from_instruction(build('phase_estimation_circuit', phase, n)).probabilities(list(range(n)))
            np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_invalid_inputs(self):
        cases = [
            ('sqrt_x_circuit', (-1,)), ('sqrt_x_circuit', (True,)),
            ('bell_state_circuit', ('bad',)), ('ghz_circuit', (1,)),
            ('phase_kickback_circuit', (float('nan'),)), ('conditional_phase_circuit', (True,)),
            ('swap_test_circuit', (0, float('inf'))), ('teleportation_circuit', ('x', 0)),
            ('increment_circuit', (0,)), ('increment_circuit', (2, .5)), ('increment_circuit', (2, 1, 4)),
            ('quantum_addition_circuit', (2, -1, 0)), ('add_squared_circuit', (2, 0, True)),
            ('conditional_increment_circuit', (2, 1, 0, 2)), ('reversible_xor_circuit', (True, 0)),
            ('grover_multi_circuit', ([],)), ('grover_multi_circuit', (['00', '00'],)),
            ('grover_multi_circuit', (['0', '11'],)), ('grover_multi_circuit', (['02'],)),
            ('grover_multi_circuit', (['00'], -1)), ('grover_multi_circuit', ('00',)),
            ('qft_signal_circuit', (2, 4)), ('qft_signal_circuit', (2, 0, 'bad')),
            ('qft_signal_circuit', (2, 1, 'square')), ('qft_circuit', (0,)),
            ('phase_estimation_circuit', (1,)), ('bernstein_vazirani_circuit', ('',)),
            ('grover_circuit', ('x',)), ('deutsch_jozsa_circuit', (2, 'bad')),
        ]
        for name, args in cases:
            with self.subTest(name=name, args=args), self.assertRaises(ValueError):
                build(name, *args)

    def test_aer_execution_smoke(self):
        # Proves the framework's measure/transpile/run path can consume the new circuits.
        backend = AerSimulator()
        for circuit, expected in [(build('quantum_addition_circuit'), '010011'),
                                  (build('qft_signal_circuit', 3, 3), '011')]:
            circuit.measure_all()
            compiled = transpile(circuit, backend, seed_transpiler=42)
            counts = backend.run(compiled, shots=64, seed_simulator=42).result().get_counts()
            self.assertEqual(counts, {expected: 64})


if __name__ == '__main__':
    unittest.main()
