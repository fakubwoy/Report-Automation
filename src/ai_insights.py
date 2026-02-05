import logging
import json
import aiohttp
import asyncio
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)


class AIInsightsEngine:
    """
    AI-Powered Manufacturing Intelligence Engine
    Uses Groq API (free, open-source models) to provide strategic insights
    """
    
    def __init__(self, api_key=None):
        """
        Initialize with Groq API
        
        Args:
            api_key: Groq API key (get free at https://console.groq.com)
                    If not provided, looks for GROQ_API_KEY environment variable
        """
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        
        # Available free models on Groq (fast inference)
        # Options: llama-3.3-70b-versatile, llama-3.1-70b-versatile, 
        #          mixtral-8x7b-32768, gemma2-9b-it
        self.model = "llama-3.3-70b-versatile"  # Best balance of speed & quality
        
        if not self.api_key:
            logging.warning("No Groq API key found. Set GROQ_API_KEY environment variable or pass api_key parameter.")
    
    async def generate_strategic_insights(self, ml_insights, production_data, kpi_summary):
        """
        Generate comprehensive strategic insights using AI
        """
        try:
            # Build comprehensive context
            context = self._build_analysis_context(ml_insights, production_data, kpi_summary)
            
            # Create detailed prompt
            prompt = f"""You are a manufacturing analytics AI expert. Analyze this production data and provide strategic insights.

MANUFACTURING DATA SUMMARY:
{json.dumps(context, indent=2)}

Please provide a comprehensive analysis with:

1. **Executive Summary** (2-3 sentences)
   - Overall operational health
   - Most critical issue requiring immediate attention

2. **Root Cause Analysis**
   - Why is downtime predicted to be {ml_insights.get('prediction', 0):.1f} minutes?
   - Which factors are contributing most?
   - Hidden patterns in the data

3. **Risk Assessment**
   - Current risk level: {ml_insights.get('risk_level', 'Unknown')}
   - Potential financial impact
   - Timeline for action

4. **Actionable Recommendations** (Priority ordered)
   - Immediate actions (next 24 hours)
   - Short-term improvements (next week)
   - Long-term optimizations

5. **Predictive Alerts**
   - What to monitor closely
   - Early warning signs to watch for

Keep your response structured, data-driven, and actionable. Focus on insights that can reduce downtime and improve quality."""

            # Call Groq API
            if not self.api_key:
                logging.warning("No API key available, using fallback insights")
                return self._generate_fallback_insights(ml_insights)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert manufacturing analytics AI that provides data-driven insights and actionable recommendations."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.7,
                        "max_tokens": 2000,
                        "top_p": 1
                    }
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        ai_analysis = data['choices'][0]['message']['content']
                        
                        logging.info(f"AI Strategic Insights generated successfully using {self.model}")
                        return self._parse_ai_response(ai_analysis)
                    else:
                        error_text = await response.text()
                        logging.warning(f"AI API call failed: {response.status} - {error_text}")
                        return self._generate_fallback_insights(ml_insights)
                        
        except Exception as e:
            logging.error(f"AI insights generation failed: {e}", exc_info=True)
            return self._generate_fallback_insights(ml_insights)
    
    async def generate_maintenance_plan(self, ml_insights, machine_stats):
        """
        Generate AI-powered maintenance schedule
        """
        try:
            prompt = f"""As a predictive maintenance AI, create a maintenance plan based on this data:

PREDICTIVE ANALYTICS:
- Predicted Downtime: {ml_insights.get('prediction', 0):.1f} minutes
- Confidence: {ml_insights.get('confidence', 0)*100:.1f}%
- Risk Level: {ml_insights.get('risk_level', 'Unknown')}
- Anomalies Detected: {ml_insights.get('anomalies_detected', 0)}

MACHINE PERFORMANCE:
{json.dumps(machine_stats, indent=2)}

Create a detailed 7-day maintenance plan with:
1. Which machines need immediate attention
2. Recommended maintenance windows
3. Parts/resources to prepare
4. Estimated downtime for each maintenance
5. Priority ranking

Format as a structured plan that can be acted upon immediately."""

            if not self.api_key:
                return "Maintenance plan generation unavailable - no API key"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a predictive maintenance expert that creates actionable maintenance schedules."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.6,
                        "max_tokens": 1500
                    }
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        maintenance_plan = data['choices'][0]['message']['content']
                        logging.info("AI Maintenance Plan generated")
                        return maintenance_plan
                    else:
                        return "Maintenance plan generation unavailable"
                        
        except Exception as e:
            logging.error(f"Maintenance plan generation failed: {e}")
            return "Maintenance plan generation unavailable"
    
    async def generate_quality_insights(self, defect_data, production_data):
        """
        Generate AI insights specifically for quality improvement
        """
        try:
            defect_rate = (defect_data.get('total_defects', 0) / 
                          defect_data.get('total_units', 1)) * 100
            
            prompt = f"""As a quality control AI expert, analyze this manufacturing quality data:

QUALITY METRICS:
- Total Units: {defect_data.get('total_units', 0):,}
- Total Defects: {defect_data.get('total_defects', 0):,}
- Defect Rate: {defect_rate:.2f}%
- Yield: {defect_data.get('yield_percentage', 0):.2f}%

PRODUCTION DETAILS:
{json.dumps(production_data, indent=2)}

Provide:
1. Quality assessment (is {defect_rate:.2f}% acceptable?)
2. Root causes of defects
3. Specific quality improvement actions
4. Target defect rate to aim for
5. Quality control checkpoints to implement

Be specific and actionable."""

            if not self.api_key:
                return "Quality insights generation unavailable - no API key"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a quality control expert specializing in manufacturing defect analysis."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.6,
                        "max_tokens": 1200
                    }
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        quality_insights = data['choices'][0]['message']['content']
                        logging.info("AI Quality Insights generated")
                        return quality_insights
                    else:
                        return "Quality insights generation unavailable"
                        
        except Exception as e:
            logging.error(f"Quality insights generation failed: {e}")
            return "Quality insights generation unavailable"
    
    def _build_analysis_context(self, ml_insights, production_data, kpi_summary):
        """Build comprehensive context for AI analysis"""
        context = {
            'timestamp': datetime.now().isoformat(),
            'ml_predictions': {
                'next_shift_downtime': ml_insights.get('prediction', 0),
                'confidence_score': ml_insights.get('confidence', 0),
                'risk_level': ml_insights.get('risk_level', 'Unknown'),
                'anomalies_detected': ml_insights.get('anomalies_detected', 0)
            },
            'feature_importance': ml_insights.get('feature_importance', {}),
            'kpi_summary': kpi_summary,
            'recommendations': ml_insights.get('recommendations', []),
            'data_quality': {
                'samples_analyzed': len(production_data) if production_data is not None else 0,
                'time_period': '7 days'
            }
        }
        
        return context
    
    def _parse_ai_response(self, ai_text):
        """Parse AI response into structured format"""
        return {
            'full_analysis': ai_text,
            'generated_at': datetime.now().isoformat(),
            'source': f'Groq AI ({self.model})',
            'sections': self._extract_sections(ai_text)
        }
    
    def _extract_sections(self, text):
        """Extract structured sections from AI response"""
        sections = {}
        
        # Simple section extraction based on numbered headers
        lines = text.split('\n')
        current_section = 'introduction'
        current_content = []
        
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.strip().split('.', 1)[1].strip().lower()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _generate_fallback_insights(self, ml_insights):
        """Generate rule-based insights when AI is unavailable"""
        prediction = ml_insights.get('prediction', 0)
        risk = ml_insights.get('risk_level', 'Unknown')
        anomalies = ml_insights.get('anomalies_detected', 0)
        
        fallback = f"""AUTOMATED INSIGHTS (AI Unavailable):

EXECUTIVE SUMMARY:
The system predicts {prediction:.1f} minutes of downtime for the next shift with a {risk.lower()} risk level. 
{"Immediate attention required." if risk in ['High', 'Critical'] else "Monitoring recommended."}

RISK ASSESSMENT:
- Current Risk Level: {risk}
- Predicted Downtime: {prediction:.1f} minutes
- Anomalies Detected: {anomalies}

RECOMMENDATIONS:
"""
        
        if risk == 'Critical':
            fallback += "1. URGENT: Schedule emergency maintenance immediately\n"
            fallback += "2. Inspect all critical systems\n"
            fallback += "3. Prepare backup equipment\n"
        elif risk == 'High':
            fallback += "1. Schedule preventive maintenance within 24 hours\n"
            fallback += "2. Monitor production closely\n"
            fallback += "3. Review recent performance logs\n"
        else:
            fallback += "1. Continue normal operations\n"
            fallback += "2. Maintain regular maintenance schedule\n"
            fallback += "3. Monitor for pattern changes\n"
        
        if anomalies > 0:
            fallback += f"\nALERT: {anomalies} anomalous patterns detected. Investigate unusual readings.\n"
        
        return {
            'full_analysis': fallback,
            'generated_at': datetime.now().isoformat(),
            'source': 'Rule-Based System',
            'sections': {'summary': fallback}
        }


async def get_comprehensive_ai_insights(ml_insights, production_data, kpi_summary, machine_stats=None, api_key=None):
    """
    Main function to get all AI-powered insights
    
    Args:
        ml_insights: ML model predictions and insights
        production_data: Historical production data
        kpi_summary: KPI metrics summary
        machine_stats: Optional machine statistics
        api_key: Optional Groq API key (otherwise uses env variable)
    """
    try:
        engine = AIInsightsEngine(api_key=api_key)
        
        # Generate strategic insights
        strategic_insights = await engine.generate_strategic_insights(
            ml_insights, production_data, kpi_summary
        )
        
        # Generate maintenance plan if machine stats available
        maintenance_plan = None
        if machine_stats:
            maintenance_plan = await engine.generate_maintenance_plan(
                ml_insights, machine_stats
            )
        
        # Generate quality insights
        quality_insights = await engine.generate_quality_insights(
            kpi_summary, production_data
        )
        
        return {
            'strategic_insights': strategic_insights,
            'maintenance_plan': maintenance_plan,
            'quality_insights': quality_insights,
            'generated_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Comprehensive AI insights failed: {e}")
        return {
            'strategic_insights': {'full_analysis': 'AI insights unavailable'},
            'maintenance_plan': 'Unavailable',
            'quality_insights': 'Unavailable',
            'generated_at': datetime.now().isoformat()
        }