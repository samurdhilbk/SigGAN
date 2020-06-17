import numpy as np
import pandas as pd
from WirelessSystem.utils import make_session_dir, store_configuration, generate_symbols, generate_rf_fingerprints, normalize, add_noise, add_non_linearity, simulate_realistic_channel

class Transmitter:
    '''
    This class contains transmitter properties
    '''
    def __init__(self, idx, x, y, nnl_cfs, authorized, impersonator=False):
        """
        x,y: coordinates
        nnl_cfs: non-linearity coefficients
        authorized: determines whether the transmitter is authorized or not 
        """
        self.x = x
        self.y = y
        self.nnl_cfs = nnl_cfs 
        self.id = idx
        self.authorized = authorized
        self.impersonator = impersonator

class RFSystem:
    '''
    Main class
    '''
    def __init__(self, n_authorized = 1, n_unauthorized = 1, mod = 'qpsk', snr = 20,
                     sess_name = 'test', area_size = 100, coef_set = 2, sess_dir = '', suffix=''):
        """
        sess_dir:   If provided, the transmitters and configuration will be loaded from the config file, otherwise they will be 
                    randomly generated. This can be used if the discriminator has already been trained
                    for this transmitter set and we want to restore it
        area_size:  Space where transmitters will be randomly placed. Locations may be used later if we implement
                    channel propagation
        coef_set:   Determines how the non-linearity coefficients will be generated by generate_rf_fingerprints(...)
        sess_name:  Used if sess_dir is not provided to name the stored directory
        mod:        Modulation order
        snr:        Signal to noise ratio
        """


        self.area_size = area_size
        self.sess_dir = sess_dir

        if sess_dir != "":
            self.auth_transmitters, self.unauth_transmitters, self.impersonator_transmitter = self.load_transmitters_from_file('./sessions/' + sess_dir, suffix=suffix)
            self.full_sess_dir='./sessions/' + sess_dir
            self.n_authorized = len(self.auth_transmitters)
            self.n_unauthorized = len(self.unauth_transmitters)
            config_file = './sessions/' + self.sess_dir + '/config%s.txt'%suffix
            config = pd.read_csv(config_file, sep = '\t').to_dict(orient = 'records')
            self.snr = ([float(record['value']) for record in config if record['parameter'] == 'snr'][0])
            self.mod = ([str(record['value']) for record in config if record['parameter'] == 'mod'][0])
            self.coef_set = ([int(record['value']) for record in config if record['parameter'] == 'coef_set'][0])

        else:
            self.snr = snr
            self.mod = mod
            self.coef_set = coef_set
            self.n_authorized = n_authorized
            self.n_unauthorized = n_unauthorized
            # Randomly generate transmitters
            self.full_sess_dir = './sessions/' + sess_name
            self.auth_transmitters = self.randomly_generate_transmitters(n_authorized, authorized = True)
            self.unauth_transmitters = self.randomly_generate_transmitters(n_unauthorized, authorized = False)
            self.impersonator_transmitter = self.randomly_generate_transmitters(1, False, True)[0]
            self.store_transmitters(self.auth_transmitters+self.unauth_transmitters+[self.impersonator_transmitter], self.full_sess_dir, suffix=suffix)
            store_configuration(self.full_sess_dir+'/config%s.txt'%suffix, mod = mod, snr = snr, coef_set = coef_set)

        # The impersonator profile is generated randomly every time
#         self.impersonator_transmitter = self.randomly_generate_transmitters(1, False)[0]

    def load_transmitters_from_file(self, sess_dir, suffix=''):
        unauth_transmitters = []
        auth_transmitters = []
        records = pd.read_csv(sess_dir+'/transmitters'+suffix+'.csv').to_dict(orient = 'records')
        for record in records:
            t = Transmitter(int(record['id']), float(record['x']), float(record['y']),
                                 [float(record['nnl_cfs_1']), float(record['nnl_cfs_2'])], 
                                     record['authorized'], record['impersonator'])    
            if record['impersonator']:
                impersonator = t
            elif record['authorized']:
                auth_transmitters.append(t)
            else:
                unauth_transmitters.append(t)
        return auth_transmitters, unauth_transmitters, impersonator

    def randomly_generate_transmitters(self, n_transmitters, authorized, impersonator=False):
        transmitters = []

        # Genrate non-linearity coeffs. Coef set 2 means random coeffs
        if n_transmitters == 1:
            coeffs = generate_rf_fingerprints(2, coef_set = self.coef_set)
        else:
            coeffs = generate_rf_fingerprints(n_transmitters, coef_set = self.coef_set)


        for i in range(n_transmitters):
            x = np.random.rand() * self.area_size
            y = np.random.rand() * self.area_size
            t = Transmitter(i, x, y, coeffs[i], authorized, impersonator)    
            transmitters.append(t)

        return transmitters

    def store_transmitters(self, transmitters, sess_dir, suffix=''):
        make_session_dir(sess_dir)

        records = {'id':[], 'x':[], 'y':[], 'nnl_cfs_1':[], 'nnl_cfs_2':[], 'authorized':[], 'impersonator':[]}
        for t in transmitters:
            records['id'].append(t.id)
            records['x'].append(t.x)
            records['y'].append(t.y)
            records['nnl_cfs_1'].append(t.nnl_cfs[0])
            records['nnl_cfs_2'].append(t.nnl_cfs[1])
            records['authorized'].append(t.authorized)
            records['impersonator'].append(t.impersonator)

        df = pd.DataFrame.from_dict(records)
        df.to_csv('%s/transmitters%s.csv' % (sess_dir, suffix), mode = 'w', index=False, header = True)

    def get_n_received_symbol_blocks(self, n_blocks, n_symbols, authorized):
        blocks = None
        txid = None
        if authorized==0:
            n_each = n_blocks // self.n_authorized
            blocks = []
            txid = []
            n_curr=0
            for tx_id in range(self.n_authorized):
                if tx_id == self.n_authorized-1: n_each = n_blocks - n_curr
                for i in range(n_each):
                    blocks.append(self.get_received_symbol_block(n_symbols, tx_id, authorized=True))
                    txid.append(tx_id)
                n_curr+=n_each
        elif authorized==1:
            n_each = n_blocks // self.n_unauthorized
            blocks = []
            txid = []
            n_curr=0
            for tx_id in range(self.n_unauthorized):
                if tx_id == self.n_unauthorized-1: n_each = n_blocks - n_curr
                for i in range(n_each):
                    blocks.append(self.get_received_symbol_block(n_symbols, tx_id, authorized=False))
                    txid.append(tx_id)
                n_curr+=n_each
        elif authorized==2:
            blocks = []
            for i in range(n_blocks):
                blocks.append(self.transmit_symbol_block(self.get_pretx_symbol_block(n_symbols), impersonator = True))
        return np.concatenate([np.expand_dims(np.real(blocks), -1),np.expand_dims(np.imag(blocks), -1)], axis=-1), txid
    
    def get_received_symbol_block(self, n_symbols, tx_id, authorized, dynamic_channel=False):
        # Generates a block of symbols for a particular transmitter 
        # from either authorized or unauthorized transmitters
        if authorized:
            t = self.auth_transmitters[tx_id]
        else:
            t = self.unauth_transmitters[tx_id]

        # Pre transmissions
        symbols = generate_symbols(n_symbols, mod=self.mod)

        # Post transmission
        symbols = add_non_linearity(symbols, t.nnl_cfs)
        symbols = normalize(symbols)
        
        if dynamic_channel==True:
            symbols = simulate_realistic_channel(symbols, self.snr)
        else:
            symbols = add_noise(symbols, self.snr)

        return symbols
    
    def get_pretx_symbol_block(self, n_symbols=128):
        # Get a block of symbols before they are transmitted 
        # and the non-linearity is created

        # Pre transmissions
        symbols = generate_symbols(n_symbols, mod=self.mod)

        return symbols

    def get_real_pretx_symbol_block(self, n_symbols=128):
        symbols = self.get_pretx_symbol_block(n_symbols)
        return np.concatenate([np.expand_dims(np.real(symbols), -1),np.expand_dims(np.imag(symbols), -1)], axis=-1)
    
    def get_n_real_pretx_symbol_blocks(self, n_blocks, n_symbols=128):
        blocks = []
        for i in range(n_blocks):
            blocks.append(self.get_real_pretx_symbol_block(n_symbols))
        return blocks
    
    def transmit_real_symbol_block(self, symbols, tx_id = 0, authorized = True, impersonator = False):
        symbols_cmplx=np.vectorize(complex)(symbols[...,0], symbols[...,1])
        transmitted_symbols_cmplx = self.transmit_symbol_block(symbols=symbols_cmplx, tx_id=tx_id, authorized=authorized, impersonator=impersonator)
        return np.concatenate([np.expand_dims(np.real(transmitted_symbols_cmplx), -1),np.expand_dims(np.imag(transmitted_symbols_cmplx), -1)], axis=-1)
    
    def transmit_symbol_block(self, symbols, tx_id = 0, authorized = True, impersonator = False, dynamic_channel=False):
        # Transmit the symbol block with particular non-linearity
        
        if impersonator:
            t = self.impersonator_transmitter
        elif authorized:
            t = self.auth_transmitters[tx_id]
        else:
            t = self.unauth_transmitters[tx_id]

        # Post transmission
        symbols = add_non_linearity(symbols, t.nnl_cfs)
        symbols = normalize(symbols)
        
        if dynamic_channel==True:
            symbols = simulate_realistic_channel(symbols, self.snr)
        else:
            symbols = add_noise(symbols, self.snr)
        
        return symbols


if __name__ == '__main__':
    rf_system = RFSystem(n_authorized = 10, n_unauthorized = 2)
    rf_system = RFSystem(sess_dir = 'test')
    rf_system.get_received_symbol_block(256, 0, authorized = True)
    rf_system.get_received_symbol_block(256, 0, authorized = False)
    rf_system.get_pretx_symbol_block(256)
    rf_system.transmit_symbol_block(256, impersonator = True)