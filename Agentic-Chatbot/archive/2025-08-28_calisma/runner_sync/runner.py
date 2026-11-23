
from prompts.global_prompt_manager import global_prompt_manager

# Create a simple Runner class to handle synchronous execution
class Runner:
    @staticmethod
    def run_sync(agent, message):
        """Synchronous wrapper for agent execution"""
        import asyncio
        try:
            # Create a simple result object
            result = asyncio.run(agent.process_message(message))
            return type('Result', (), {'final_output': result})()
        except Exception as e:
            print(f"Error running agent: {e}")
            return type('Result', (), {'final_output': message})()
        
        

def correct_user_query(user_query):
    """
    Kullanıcı sorgusunu düzeltir ve anlamsal olarak en yakın düzgün cümleye dönüştürür
    """
    try:
        query_correction_agent = global_prompt_manager.get("query_correction_agent_system_prompt")
        result = Runner.run_sync(query_correction_agent, user_query)
        corrected_query = result.final_output.strip()
        
        # Eğer düzeltme yapıldıysa kullanıcıya bilgi ver
        if corrected_query != user_query:
            print(f"🔧 Sorgu düzeltildi:")
            print(f"   Orijinal: '{user_query}'")
            print(f"   Düzeltilmiş: '{corrected_query}'")
            print(f"   Merhabalar! Kast ettiğiniz tur içeriği '{corrected_query}' şeklinde anlaşılmıştır. Sizler için kontrol ediyorum...")
        
        return corrected_query
    except Exception as e:
        print(f"⚠️ Sorgu düzeltme hatası: {str(e)}")
        return user_query

def analyze_tour_query(user_query):
    """
    Kullanıcı sorgusunu analiz eder ve tur alanlarıyla eşleştirir
    """
    try:
        analysis_agent = global_prompt_manager.get("tour_analysis_agent_system_prompt")
        result = Runner.run_sync(analysis_agent, user_query)
        return result.final_output
    except Exception as e:
        return f"Analiz hatası: {str(e)}"