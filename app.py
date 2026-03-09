import traceback

from flask import Flask, jsonify, render_template, request

from config import config

# 这里先占位导入，后续会在 agents 包中实现具体逻辑
from agents.validation_agent import run_validation_agent
from agents.query_agent import run_query_agent


def create_app() -> Flask:
    """
    创建并配置 Flask 应用实例。

    说明：
    - 为了便于后续扩展与测试，使用工厂函数模式创建应用。
    """
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        """
        渲染前端首页。
        """
        return render_template("index.html")

    @app.route("/chat", methods=["POST"])
    def chat():
        """
        通用聊天接口，用于承载两种场景：
        - scene = validation：数据验证场景
        - scene = query：查数找数场景
        """
        data = request.get_json(force=True) or {}
        message = data.get("message", "").strip()
        scene = data.get("scene", "validation")

        if not message:
            return jsonify({"error": "message 不能为空"}), 400

        try:
            if scene == "validation":
                # 调用数据验证 Agent
                response_text = run_validation_agent(user_input=message)
            elif scene == "query":
                # 调用查数找数 Agent（包含 Text-to-SQL + RAG）
                response_text = run_query_agent(user_input=message)
            else:
                return jsonify({"error": f"未知场景: {scene}"}), 400

            return jsonify({"message": response_text})
        except Exception as e:
            # 返回详细错误信息，便于在聊天框中排查（含 traceback 与 API 响应体）
            err_msg = str(e)
            tb = traceback.format_exc()
            detail = {"traceback": tb}
            if getattr(e, "response", None) is not None:
                try:
                    detail["response_status"] = getattr(e.response, "status_code", None)
                    detail["response_body"] = getattr(e.response, "text", None) or (e.response.json() if hasattr(e.response, "json") else None)
                except Exception:
                    detail["response_body"] = str(getattr(e, "body", ""))
            return jsonify({"error": err_msg, "detail": detail}), 500

    return app


if __name__ == "__main__":
    # 仅用于本地开发调试，生产环境建议使用 WSGI / ASGI 容器启动
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=config.FLASK_DEBUG)

