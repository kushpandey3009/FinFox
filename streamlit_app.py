import streamlit as st
import yfinance as yf
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class StockDataAssistant:
    def __init__(self, openai_api_key: str):
        """
        Initialize the stock data assistant with OpenAI and yfinance
        
        :param openai_api_key: OpenAI API key for natural language processing
        """
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize OpenAI Language Model
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.3,
            max_tokens=200
        )
        
        # Create a comprehensive prompt template for intelligent ticker extraction
        self.ticker_extraction_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are FinFox, an expert financial analyst specializing in Indian stocks, 
            with a keen eye for identifying company tickers.

            Your mission: Extract the most precise stock ticker for Indian companies.

            Query: "{query}"

            Extraction Guidelines:
            1. Identify the specific Indian company mentioned
            2. Extract the most accurate stock ticker
            3. Consider:
               - Full company names
               - Common abbreviations
               - Context of the query
            
            Precise Extraction Rules:
            - Prioritize Indian stock exchanges (NSE/BSE)
            - Be extremely specific about the company
            - If multiple possibilities exist, choose the most prominent
            - If no clear stock-related query, return 'NON_STOCK'

            Output Format:
            - Return ONLY the ticker symbol in UPPERCASE
            - For Indian stocks, use NSE ticker
            - If uncertain, return 'NONE'
            """
        )
        
        # Create LLM chain for ticker extraction
        self.ticker_extraction_chain = LLMChain(
            llm=self.llm, 
            prompt=self.ticker_extraction_prompt
        )

    def extract_ticker(self, query: str):
        """
        Intelligently extract stock ticker
        
        :param query: User's input query
        :return: Stock ticker or None
        """
        try:
            # Extract ticker using LLM
            ticker_response = self.ticker_extraction_chain.run(query=query).strip().upper()
            
            # Handle different scenarios
            if ticker_response in ['NON_STOCK', 'NONE', '']:
                return None
            
            # Append .NS for NSE exchange
            return f"{ticker_response}.NS"
        
        except Exception as e:
            st.error(f"Error in ticker extraction: {e}")
            return None

    def fetch_stock_data(self, ticker: str):
        """
        Fetch comprehensive stock data for a given ticker
        
        :param ticker: Stock ticker symbol
        :return: Dictionary of stock information
        """
        try:
            # Create Ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch current stock info
            info = stock.info
            
            # Get historical data (1 month)
            history = stock.history(period='1mo')
            
            # Prepare result dictionary with comprehensive information
            result = {
                'Ticker': ticker,
                'Company Name': info.get('longName', 'N/A'),
                'Current Price': f"₹{info.get('currentPrice', 'N/A'):,.2f}",
                'Previous Close': f"₹{info.get('previousClose', 'N/A'):,.2f}",
                'Market Cap': self._format_market_cap(info.get('marketCap', 'N/A')),
                '52 Week High': f"₹{info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}",
                '52 Week Low': f"₹{info.get('fiftyTwoWeekLow', 'N/A'):,.2f}",
                'Today\'s Low': f"₹{history['Low'].min():,.2f}" if not history.empty else 'N/A',
                'Today\'s High': f"₹{history['High'].max():,.2f}" if not history.empty else 'N/A',
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                'Dividend Yield': f"{info.get('dividendYield', 'N/A')*100:.2f}%" if isinstance(info.get('dividendYield'), float) else 'N/A',
                'EPS': info.get('trailingEps', 'N/A')
            }
            
            return result
        
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None

    def _format_market_cap(self, market_cap):
        """
        Format market cap to readable format
        
        :param market_cap: Market cap value
        :return: Formatted market cap string
        """
        if not isinstance(market_cap, (int, float)):
            return market_cap
        
        if market_cap >= 1_000_000_000_000:
            return f"₹{market_cap/1_000_000_000_000:.2f} Trillion"
        elif market_cap >= 1_000_000_000:
            return f"₹{market_cap/1_000_000_000:.2f} Billion"
        elif market_cap >= 1_000_000:
            return f"₹{market_cap/1_000_000:.2f} Million"
        else:
            return f"₹{market_cap:,.2f}"

def main():
    # Set page configuration
    st.set_page_config(
        page_title="FinFox - Indian Stock Assistant",
        page_icon="🦊",
        layout="centered"
    )

    # Title and description
    st.title("🦊 FinFox: Indian Stock Information Assistant")
    st.markdown("Get instant insights into Indian stock market companies!")

    # OpenAI API Key input (in a real-world app, use secure environment variables)
    openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

    # Query input
    query = st.text_input("Enter company name or stock query", placeholder="e.g., Infosys, TCS, Reliance")

    # Initialize stock assistant when API key is provided
    if openai_api_key:
        try:
            # Create stock data assistant
            assistant = StockDataAssistant(openai_api_key)

            # Search button
            if st.button("Search Stock"):
                if query:
                    # Extract ticker
                    ticker = assistant.extract_ticker(query)

                    if ticker:
                        # Fetch stock data
                        stock_data = assistant.fetch_stock_data(ticker)

                        if stock_data:
                            # Display stock information
                            st.header(f"{stock_data['Company Name']} Stock Details")
                            
                            # Create two columns for layout
                            col1, col2 = st.columns(2)

                            # Display stock details in a grid-like format
                            for key, value in stock_data.items():
                                if key not in ['Ticker', 'Company Name']:
                                    with col1 if list(stock_data.keys()).index(key) % 2 == 0 else col2:
                                        st.metric(label=key, value=value)
                        else:
                            st.warning("Could not retrieve stock information.")
                    else:
                        st.error("Could not identify a stock ticker for the given query.")
                else:
                    st.warning("Please enter a company name or stock query.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please enter your OpenAI API Key in the sidebar to use FinFox.")

    # Footer
    st.markdown("---")
    st.markdown("💡 FinFox helps you quickly retrieve stock information for Indian companies")

if __name__ == "__main__":
    main()