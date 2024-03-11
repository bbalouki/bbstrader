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
        'D1':  1536
    }
    return time_frame_mapping


def init_trade(args, symbol=None):
    # Initializes and returns a Trade object
    if symbol is not None:
        trade_symbol = symbol
    else : trade_symbol = args.symbol
    trade = Trade(
        expert_name=args.expert_name,
        version=args.version,
        expert_id=args.expert_id,
        symbol=trade_symbol,
        max_risk=args.max_risk,
        target=args.target,
        daily_risk=args.daily_risk,
        max_trades=args.max_trades,
        account_leverage=args.account_leverage,
        time_frame=args.time_frame,
        start_time=args.start_time,
        finishing_time=args.finishing_time,
        ending_time=args.ending_time,
        std_stop=args.std_stop,
        pchange_sl=args.pchange_sl,
        sl=args.sl,
        tp=args.tp,
        be=args.be,
        rr=args.rr
    )
    return trade

def add_trade_arguments(parser, pair=False, pchange_sl=None, strategy=None):
    # Add arguments for Trade parameters
    parser.add_argument('--expert_name', type=str, default='bbstrader', help='Expert name')
    parser.add_argument('--expert_id', type=int, default=0, help='Expert ID')
    parser.add_argument('--version', type=float, default=1.0, help='Version')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol to trade')
    parser.add_argument('--max_risk', type=float, default=10.0, help='Maximum risk')
    parser.add_argument('--target', type=float, default=2.0, help='Target')
    parser.add_argument(
        '--daily_risk', type=float, default=0.25, help='Daily risk')
    parser.add_argument(
        '--max_trades', type=int, default=None, help='Maximum trades', nargs='?', const=None)
    parser.add_argument('--account_leverage', type=bool, default=True, help='Account leverage')
    if strategy is not None:
        if strategy == 'sma':
            parser.add_argument('--time_frame', type=str, default='1h', help='Time frame')
        elif strategy == 'pair':
            parser.add_argument('--time_frame', type=str, default='D1', help='Time frame')
        elif strategy == 'ou':
            parser.add_argument('--time_frame', type=str, default='1h', help='Time frame')
        elif strategy == 'arch':
            parser.add_argument('--time_frame', type=str, default='D1', help='Time frame')
    else: parser.add_argument('--time_frame', type=str, required=True, help='Time frame')
    parser.add_argument('--start_time', type=str, default='14:35', help='Start time')
    parser.add_argument('--finishing_time', type=str, default='20:50', help='Finishing time')
    parser.add_argument('--ending_time', type=str, default='20:55', help='Ending time')
    parser.add_argument('--std_stop', type=bool, default=False, help='Standard stop')
    parser.add_argument('--rr', type=float, default=3.0, help='Risk reward ratio')
    if pair:
        if pchange_sl is not None:
            parser.add_argument(
                '--pchange_sl', type=float, default=pchange_sl, help='Percentage change stop loss')
        else: parser.add_argument(
            '--pchange_sl', type=float, default=2.5, help='Percentage change stop loss')
    else: parser.add_argument(
        '--pchange_sl', type=float, default=None, help='Percentage change stop loss', nargs='?', const=None)
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
    parser.add_argument('--max_t', type=int, default=1, help='Maximum trades')
    parser.add_argument('--iter_time', type=float,default=60, nargs='?', help='Iteration time')
    parser.add_argument('--risk_manager', type=str, default='hmm', help='Risk manager')
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
    return parser

def add_pair_trading_arguments(parser, pair=True, pchange_sl=2.5):
    # Add arguments for Trade parameters
    parser = add_trade_arguments(parser, pair=pair, pchange_sl=pchange_sl, strategy='pair')
    # Add common trading arguments
    parser = add_common_trading_arguments(parser, strategy='pair')
    # Add arguments for pair_trading parameters
    parser.add_argument(
        '--pair', nargs=2, required=True, help='Pair of symbols for trading, e.g., --pair GOOG MSFT')
    parser.add_argument('--ols', type=int, default=50, help='OLS period')
    parser.add_argument('--hmm_ticker', type=str, help='HMM ticker')

    return parser

def add_ou_trading_arguments(parser):
    # Add arguments for Trade parameters
    parser = add_trade_arguments(parser, strategy='ou')
    # Add common trading arguments
    parser = add_common_trading_arguments(parser, strategy='ou')
    # Add arguments for ou_trading parameters
    parser.add_argument('--p', type=int, default=20, help='Number of time steps')
    parser.add_argument('--n', type=int, default=20, help='Number of simulations to perform')
    parser.add_argument('--ou_window', type=int, default=2000, nargs='?', help='OU window size ')
    parser.add_argument('--hmm_window', type=int, default=60, nargs='?', help='HMM window size ')
    return parser

def add_arch_trading_arguments(parser):
    # Add arguments for Trade parameters
    parser = add_trade_arguments(parser, strategy='arch')
    # Add common trading arguments
    parser = add_common_trading_arguments(parser, strategy='arch')
    # Add arguments for ou_trading parameters
    parser.add_argument('--k', type=int, default=500, help='K parameter for the strategy')
    parser.add_argument('--hmm_window', type=int, default=50, nargs='?', help='HMM window size ')
    return parser