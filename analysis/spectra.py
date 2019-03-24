import pymaster as nmt
import numpy as np


class Spectrum(object):
    def __init__(self, name1, name2, leff, nell, cell, windows):
        self.names = (name1, name2)
        self.leff = leff
        self.nell = nell
        self.cell = cell
        self.windows = windows

    @classmethod
    def from_fields(Spectrum, field1, field2, bpws,
                    wsp=None, save_windows=True):
        leff = bpws.bn.get_effective_ells()

        # Compute MCM if needed
        if wsp is None:
            wsp = nmt.NmtWorkspace()
            wsp.compute_coupling_matrix(field1.field,
                                        field2.field,
                                        bpws.bn)

        # Compute data power spectrum
        cell = wsp.decouple_cell(nmt.compute_coupled_cell(field1.field,
                                                          field2.field))[0]

        # Compute noise power spectrum if needed
        if field1.is_ndens and field2.is_ndens and field1.name == field2.name:
            nl = wsp.couple_cell([np.ones(3 * field1.nside) / field1.ndens])
            nell = wsp.decouple_cell(nl)[0]
        else:
            nell = np.zeros(len(leff))

        # Compute bandpower windows
        nbpw = len(cell)
        lmax = 3*field1.nside-1
        if save_windows:
            windows = np.zeros([nbpw, lmax+1])
            for il in range(lmax+1):
                t_hat = np.zeros(lmax+1)
                t_hat[il] = 1.
                windows[:, il] = wsp.decouple_cell(wsp.couple_cell([t_hat]))
        else:
            windows = None
        return Spectrum(field1.name, field2.name, leff, nell, cell, windows)

    @classmethod
    def from_file(Spectrum, fname, name1, name2):
        d = np.load(fname)
        return Spectrum(name1, name2, d['ls'], d['nls'], d['cls'],
                        d['windows'])

    def to_file(self, fname):
        np.savez(fname[:-4],  # Remove file suffix
                 ls=self.leff, cls=self.cell,
                 nls=self.nell, windows=self.windows)
