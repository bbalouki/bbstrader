from mtrader5.trade import Trade

def tf_mapping():
    time_frame_mapping = {
        '1m':  1,   
        '3m':  3,    
        '5m':  5,    
        '10m': 10,   
        '15m': 15,   
        '30m': 30,   
        '1h':  60,   
        '2h':  120,  
        '4h':  240,  
        'D1':  1440
    }
    return time_frame_mapping


def init_trade(args, symbol=None):
    # Initializes and returns a Trade object
    if symbol is not None:
        trade_symbol = symbol
    else : trade_symbol = args.symbol
    trade = Trade(
        expert_name=args.expert,
        version=args.version,
        expert_id=args.id,
        symbol=trade_symbol,
        max_risk=args.mr,
        target=args.t,
        daily_risk=args.dr,
        max_trades=args.maxt,
        account_leverage=args.acl,
        time_frame=args.tfm,
        start_time=args.start,
        finishing_time=args.fint,
        ending_time=args.endt,
        std_stop=args.std,
        pchange_sl=args.psl,
        sl=args.sl,
        tp=args.tp,
        be=args.be,
        rr=args.rr
    )
    return trade

def add_trade_arguments(parser, pair=False, pchange_sl=None, strategy=None):
    # Add arguments for Trade parameters
    parser.add_argument('--expert', type=str, default='bbstrader', help='Expert name')
    parser.add_argument('--id', type=int, default=0, help='Expert ID')
    parser.add_argument('--version', type=float, default=1.0, help='Version')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to trade')
    parser.add_argument('--mr', type=float, default=5.0, help='Maximum risk')
    parser.add_argument('--t', type=float, default=2.0, help='Target')
    parser.add_argument(
        '--dr', type=float, default=0.25, help='Daily risk')
    parser.add_argument(
        '--maxt', type=int, default=20, help='Maximum trades', nargs='?', const=None)
    parser.add_argument('--acl', type=bool, default=True, help='Account leverage')
    if strategy is not None:
        if strategy == 'sma':
            parser.add_argument('--tfm', type=str, default='1h', help='Time frame')
        elif strategy == 'pair':
            parser.add_argument('--tfm', type=str, default='D1', help='Time frame')
        elif strategy == 'ou':
            parser.add_argument('--tfm', type=str, default='1h', help='Time frame')
        elif strategy == 'arch':
            parser.add_argument('--tfm', type=str, default='D1', help='Time frame')
    else: parser.add_argument('--tfm', type=str, required=True, help='Time frame')
    parser.add_argument('--start', type=str, default='13:35', help='Start time')
    parser.add_argument('--fint', type=str, default='19:50', help='Finishing time')
    parser.add_argument('--endt', type=str, default='19:55', help='Ending time')
    parser.add_argument('--std', type=bool, default=False, help='Standard Deviation stop')
    parser.add_argument('--rr', type=float, default=3.0, help='Risk reward ratio')
    if pair:
        if pchange_sl is not None:
            parser.add_argument(
                '--psl', type=float, default=pchange_sl, help='Percentage change stop loss')
        else: parser.add_argument(
            '--psl', type=float, default=2.5, help='Percentage change stop loss')
    else: parser.add_argument(
        '--psl', type=float, default=None, help='Percentage change stop loss', nargs='?', const=None)
    parser.add_argument('--sl', type=int, default=None, help='Stop loss', nargs='?', const=None)
    parser.add_argument('--tp', type=int, default=None, help='Take profit', nargs='?', const=None)
    parser.add_argument('--be', type=int, default=None, help='Break even', nargs='?', const=None)
    return parser

def add_common_trading_arguments(parser, strategy=None):
    # Common trading arguments
    if strategy is not None:
        if strategy == 'sma':
            parser.add_argument('--tf', type=str, default='1h', help='Time frame for trading')
        elif strategy == 'pair':
            parser.add_argument('--tf', type=str, default='D1', help='Time frame for trading')
        elif strategy == 'ou':
            parser.add_argument('--tf', type=str, default='1h', help='Time frame for trading')
        elif strategy == 'arch':
            parser.add_argument('--tf', type=str, default='D1', help='Time frame for trading')
    else: parser.add_argument('--tf', type=str, required=True, help='Time frame for trading')
    parser.add_argument('--mm', type=bool, default=True, help='Money management')
    parser.add_argument('--mxt', type=int, default=1, help='Maximum trades')
    parser.add_argument('--it', type=float,default=30, help='Iteration time')
    parser.add_argument('--period', type=str, default='week', choices=['day', 'week', 'month'], help='Trading period')
    return parser

def add_sma_trading_arguments(parser):
    # Add arguments for Trade parameters
    parser = add_trade_arguments(parser, strategy='sma')
    # Add common trading arguments
    parser = add_common_trading_arguments(parser, strategy='sma')
    # Add arguments for sma_trading parameters
    parser.add_argument('--sma', type=int, default=35, help='Short moving average period')
    parser.add_argument('--lma', type=int, default=80, help='Long moving average period')
    parser.add_argument('--rm', type=str, default='hmm', help='Risk manager')
    return parser

def add_pair_trading_arguments(parser, pair=True, pchange_sl=2.5):
    # Add arguments for Trade parameters
    parser = add_trade_arguments(parser, pair=pair, pchange_sl=pchange_sl, strategy='pair')
    # Add common trading arguments
    parser = add_common_trading_arguments(parser, strategy='pair')
    # Add arguments for pair_trading parameters
    parser.add_argument(
        '--pair', nargs=2, required=True, help='Pair of symbols for trading, e.g., --pair GOOG MSFT')
    parser.add_argument('--rm', type=str, default=None, help='Risk manager', nargs='?', const=None)
    parser.add_argument('--rmt', type=str, default=None, help='Risk manager ticker', nargs='?', const=None)
    parser.add_argument('--rmw', type=int, default=None, help='Risk manager Window', nargs='?', const=None)
    return parser

def add_ou_trading_arguments(parser):
    # Add arguments for Trade parameters
    parser = add_trade_arguments(parser, strategy='ou')
    # Add common trading arguments
    parser = add_common_trading_arguments(parser, strategy='ou')
    # Add arguments for ou_trading parameters
    parser.add_argument('--p', type=int, default=20, help='Number of time steps')
    parser.add_argument('--n', type=int, default=20, help='Number of simulations to perform')
    parser.add_argument('--ouw', type=int, default=2000, help='OU window size parameters estimation')
    parser.add_argument('--rmw', type=int, default=60, help='Risk manager Window', nargs='?', const=None)
    parser.add_argument('--rm', type=str, default=None, help='Risk manager', nargs='?', const=None)
    return parser

def add_arch_trading_arguments(parser):
    # Add arguments for Trade parameters
    parser = add_trade_arguments(parser, strategy='arch')
    # Add common trading arguments
    parser = add_common_trading_arguments(parser, strategy='arch')
    # Add arguments for ou_trading parameters
    parser.add_argument('--k', type=int, default=500, help='Window used for prediction')
    parser.add_argument('--rmw', type=int, default=60, help='Risk manager Window', nargs='?', const=None)
    parser.add_argument('--rm', type=str, default=None, help='Risk manager', nargs='?', const=None)
    return parser