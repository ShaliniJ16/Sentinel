import { Router, type IRouter, type Request } from "express";
import cookieParser from "cookie-parser";
import { and, asc, avg, count, desc, eq, sql } from "drizzle-orm";
import {
  AnalyzeTrafficBody,
  AnalyzeTrafficResponse,
  AuditLog,
  GetCurrentUserResponse,
  GetDashboardResponse,
  GetPrivacyResponse,
  ListAuditLogsResponse,
  ListIntrusionEventsResponse,
  ListUsersResponse,
  LoginBody,
  LoginResponse,
  LogoutResponse,
  RegisterBody,
  RegisterResponse,
  UpdatePrivacyBody,
  UpdatePrivacyResponse,
  UpdateUserRoleBody,
  UpdateUserRoleParams,
  UpdateUserRoleResponse,
  VerifyAuditLogsResponse,
} from "@workspace/api-zod";
import {
  auditLogsTable,
  db,
  intrusionEventsTable,
  privacySettingsTable,
  sessionsTable,
  usersTable,
} from "@workspace/db";
import { isolationForestScore, riskForAnomaly, type TrafficVector } from "../lib/isolation-forest";
import {
  appendAuditLog,
  cookieOptions,
  createSession,
  deleteSession,
  epsilonFor,
  ensureDemoData,
  getPrivacyLevel,
  getUserBySessionToken,
  hashPassword,
  laplaceNoise,
  maskEmail,
  maskIp,
  maskUsername,
  riskFromTrust,
  trustScoreFor,
  verifyAuditChain,
  verifyPassword,
  type PrivacyLevel,
  type Role,
} from "../lib/security";

const router: IRouter = Router();

function sessionToken(req: Request): string | undefined {
  return req.cookies?.security_session as string | undefined;
}

async function currentUser(req: Request) {
  return getUserBySessionToken(sessionToken(req));
}

function sessionUser(user: typeof usersTable.$inferSelect) {
  const trustScore = trustScoreFor(user);
  return {
    id: user.id,
    username: user.username,
    maskedEmail: maskEmail(user.email),
    role: user.role as Role,
    trustScore,
    riskLevel: riskFromTrust(trustScore),
    knownDevice: user.knownDevice,
  };
}

function iso(value: Date | null): string | null {
  return value ? value.toISOString() : null;
}

function userRecord(user: typeof usersTable.$inferSelect) {
  const summary = sessionUser(user);
  return {
    ...summary,
    failedLogins: user.failedLogins,
    lastLogin: iso(user.lastLogin),
    status: user.failedLogins >= 3 ? "Review" : "Active",
  } as const;
}

function errorMessage(res: Parameters<IRouter["post"]>[1] extends never ? never : any, status: number, message: string) {
  res.status(status).json({ error: message });
}

async function requireUser(req: Request, res: any) {
  const user = await currentUser(req);
  if (!user) {
    res.status(401).json({ error: "Sign in to continue." });
    return null;
  }
  await ensureDemoData();
  return user;
}

router.post("/auth/register", async (req, res): Promise<void> => {
  const parsed = RegisterBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }
  const username = parsed.data.username.trim();
  const email = parsed.data.email.toLowerCase().trim();
  const [existing] = await db.select().from(usersTable).where(eq(usersTable.email, email)).limit(1);
  if (existing) {
    res.status(400).json({ error: "An account with that email already exists." });
    return;
  }
  const [{ userCount }] = await db.select({ userCount: count() }).from(usersTable);
  const role: Role = Number(userCount) === 0 ? "Admin" : "User";
  const [user] = await db.insert(usersTable).values({
    username,
    email,
    passwordHash: hashPassword(parsed.data.password),
    role,
    trustScore: 100,
    failedLogins: 0,
    knownDevice: true,
  }).returning();
  await db.insert(privacySettingsTable).values({ userId: user.id, level: "Medium" });
  const token = await createSession(user.id);
  res.cookie("security_session", token, cookieOptions());
  await appendAuditLog({
    userId: user.id,
    eventType: "Account created",
    description: `${username} joined the security workspace.`,
    riskScore: 0.08,
    actor: username,
  });
  res.status(201).json(RegisterResponse.parse(sessionUser(user)));
});

router.post("/auth/login", async (req, res): Promise<void> => {
  const parsed = LoginBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }
  const email = parsed.data.email.toLowerCase().trim();
  const [user] = await db.select().from(usersTable).where(eq(usersTable.email, email)).limit(1);
  if (!user || !verifyPassword(parsed.data.password, user.passwordHash)) {
    if (user) {
      const failedLogins = user.failedLogins + 1;
      await db.update(usersTable).set({ failedLogins, trustScore: trustScoreFor({ ...user, failedLogins }) }).where(eq(usersTable.id, user.id));
      await appendAuditLog({
        userId: user.id,
        eventType: "Failed login",
        description: `Failed sign-in attempt for ${maskEmail(user.email)}.`,
        riskScore: Math.min(1, failedLogins / 5),
        actor: "Unknown session",
      });
    }
    res.status(401).json({ error: "Email or password is incorrect." });
    return;
  }
  const [updated] = await db.update(usersTable).set({
    failedLogins: 0,
    lastLogin: new Date(),
    knownDevice: true,
    trustScore: trustScoreFor({ failedLogins: 0, knownDevice: true }),
  }).where(eq(usersTable.id, user.id)).returning();
  const token = await createSession(updated.id);
  res.cookie("security_session", token, cookieOptions());
  await appendAuditLog({
    userId: updated.id,
    eventType: "Successful login",
    description: `${updated.username} authenticated with a trusted device.`,
    riskScore: 0.12,
    actor: updated.username,
  });
  res.json(LoginResponse.parse(sessionUser(updated)));
});

router.post("/auth/logout", async (req, res): Promise<void> => {
  await deleteSession(sessionToken(req));
  res.clearCookie("security_session", { path: "/" });
  res.json(LogoutResponse.parse({ message: "Signed out securely." }));
});

router.get("/auth/me", async (req, res): Promise<void> => {
  const user = await currentUser(req);
  if (!user) {
    res.status(401).json({ error: "Not authenticated." });
    return;
  }
  res.json(GetCurrentUserResponse.parse(sessionUser(user)));
});

router.get("/users", async (req, res): Promise<void> => {
  const actor = await requireUser(req, res);
  if (!actor) return;
  const rows = await db.select().from(usersTable).orderBy(desc(usersTable.createdAt));
  res.json(ListUsersResponse.parse(rows.map(userRecord)));
});

router.patch("/users/:id/role", async (req, res): Promise<void> => {
  const actor = await requireUser(req, res);
  if (!actor) return;
  if (actor.role !== "Admin") {
    res.status(403).json({ error: "Only an Admin can change roles." });
    return;
  }
  const params = UpdateUserRoleParams.safeParse(req.params);
  const parsed = UpdateUserRoleBody.safeParse(req.body);
  if (!params.success || !parsed.success) {
    res.status(400).json({ error: "Enter a valid user id and role." });
    return;
  }
  const [updated] = await db.update(usersTable).set({ role: parsed.data.role }).where(eq(usersTable.id, params.data.id)).returning();
  if (!updated) {
    res.status(404).json({ error: "User not found." });
    return;
  }
  await appendAuditLog({
    userId: actor.id,
    eventType: "Role updated",
    description: `${updated.username} is now ${updated.role}.`,
    riskScore: 0.2,
    actor: actor.username,
  });
  res.json(UpdateUserRoleResponse.parse(userRecord(updated)));
});

async function privacyPayload(userId: number) {
  const level = await getPrivacyLevel(userId);
  const epsilon = epsilonFor(level);
  const [{ threatCount }] = await db.select({ threatCount: count() }).from(intrusionEventsTable).where(eq(intrusionEventsTable.classification, "Suspicious"));
  const [{ averageTrust }] = await db.select({ averageTrust: avg(usersTable.trustScore) }).from(usersTable);
  const rawThreats = Number(threatCount);
  const rawAverageTrust = Number(averageTrust ?? 0);
  const scale = 1 / epsilon;
  return {
    level,
    status: `${level} privacy protection active`,
    epsilon,
    maskedEmail: maskEmail("analyst@northstar.security"),
    maskedUsername: maskUsername("northstar-analyst"),
    maskedIp: maskIp("192.168.40.18"),
    rawThreats,
    privateThreats: Math.max(0, Number((rawThreats + laplaceNoise(scale)).toFixed(1))),
    rawAverageTrust: Number(rawAverageTrust.toFixed(1)),
    privateAverageTrust: Number(Math.max(0, Math.min(100, rawAverageTrust + laplaceNoise(scale * 4))).toFixed(1)),
  };
}

router.get("/privacy", async (req, res): Promise<void> => {
  const actor = await requireUser(req, res);
  if (!actor) return;
  res.json(GetPrivacyResponse.parse(await privacyPayload(actor.id)));
});

router.patch("/privacy", async (req, res): Promise<void> => {
  const actor = await requireUser(req, res);
  if (!actor) return;
  const parsed = UpdatePrivacyBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }
  await db.insert(privacySettingsTable).values({ userId: actor.id, level: parsed.data.level }).onConflictDoUpdate({
    target: privacySettingsTable.userId,
    set: { level: parsed.data.level, updatedAt: new Date() },
  });
  await appendAuditLog({
    userId: actor.id,
    eventType: "Privacy level changed",
    description: `${actor.username} selected ${parsed.data.level} privacy.`,
    riskScore: parsed.data.level === "High" ? 0.05 : 0.1,
    actor: actor.username,
  });
  res.json(UpdatePrivacyResponse.parse(await privacyPayload(actor.id)));
});

function trafficVector(input: {
  duration: number;
  srcBytes: number;
  dstBytes: number;
  failedLogins: number;
  packetCount: number;
  connectionRate: number;
}): TrafficVector {
  return [
    input.duration,
    Math.log1p(input.srcBytes),
    Math.log1p(input.dstBytes),
    input.failedLogins,
    input.packetCount,
    input.connectionRate,
  ];
}

async function intrusionPayload(row: typeof intrusionEventsTable.$inferSelect) {
  return {
    id: row.id,
    duration: row.duration,
    srcBytes: row.srcBytes,
    dstBytes: row.dstBytes,
    failedLogins: row.failedLogins,
    packetCount: row.packetCount,
    connectionRate: row.connectionRate,
    classification: row.classification as "Normal" | "Suspicious",
    anomalyScore: row.anomalyScore,
    riskLevel: row.riskLevel as "Low" | "Medium" | "High",
    timestamp: row.timestamp.toISOString(),
  };
}

router.post("/intrusion/analyze", async (req, res): Promise<void> => {
  const actor = await requireUser(req, res);
  if (!actor) return;
  const parsed = AnalyzeTrafficBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: parsed.error.message });
    return;
  }
  const history = await db.select().from(intrusionEventsTable).orderBy(desc(intrusionEventsTable.timestamp)).limit(96);
  const input = parsed.data;
  const score = isolationForestScore(
    trafficVector(input),
    history.map(trafficVector),
  );
  const riskLevel = riskForAnomaly(score);
  const classification = score >= 0.62 ? "Suspicious" : "Normal";
  const [row] = await db.insert(intrusionEventsTable).values({
    ...input,
    userId: actor.id,
    anomalyScore: Number(score.toFixed(3)),
    classification,
    riskLevel,
  }).returning();
  await appendAuditLog({
    userId: actor.id,
    eventType: classification === "Suspicious" ? "Threat detected" : "Traffic analyzed",
    description: `${classification} network flow scored ${(score * 100).toFixed(0)}%.`,
    riskScore: score,
    actor: actor.username,
  });
  res.json(AnalyzeTrafficResponse.parse(await intrusionPayload(row)));
});

router.get("/intrusion/events", async (req, res): Promise<void> => {
  const actor = await requireUser(req, res);
  if (!actor) return;
  const rows = await db.select().from(intrusionEventsTable).orderBy(desc(intrusionEventsTable.timestamp)).limit(50);
  res.json(ListIntrusionEventsResponse.parse(await Promise.all(rows.map(intrusionPayload))));
});

router.get("/audit-logs", async (req, res): Promise<void> => {
  const actor = await requireUser(req, res);
  if (!actor) return;
  const rows = await db.select().from(auditLogsTable).orderBy(desc(auditLogsTable.id)).limit(80);
  res.json(ListAuditLogsResponse.parse(rows.map((row) => ({
    id: row.id,
    timestamp: row.timestamp.toISOString(),
    eventType: row.eventType,
    riskScore: row.riskScore,
    previousHash: row.previousHash,
    currentHash: row.currentHash,
    actor: row.actor,
  }))));
});

router.post("/audit-logs/verify", async (req, res): Promise<void> => {
  const actor = await requireUser(req, res);
  if (!actor) return;
  const result = await verifyAuditChain();
  res.json(VerifyAuditLogsResponse.parse(result));
});

router.get("/dashboard", async (req, res): Promise<void> => {
  const actor = await requireUser(req, res);
  if (!actor) return;
  const [userTotals] = await db.select({ totalUsers: count(), averageTrustScore: avg(usersTable.trustScore) }).from(usersTable);
  const [threatTotals] = await db.select({ threatsDetected: count() }).from(intrusionEventsTable).where(eq(intrusionEventsTable.classification, "Suspicious"));
  const [highTotals] = await db.select({ highRiskEvents: count() }).from(intrusionEventsTable).where(eq(intrusionEventsTable.riskLevel, "High"));
  const threatRows = await db.select({ classification: intrusionEventsTable.classification, total: count() }).from(intrusionEventsTable).groupBy(intrusionEventsTable.classification);
  const riskRows = await db.select({ riskLevel: intrusionEventsTable.riskLevel, total: count() }).from(intrusionEventsTable).groupBy(intrusionEventsTable.riskLevel);
  const activityRows = await db.select().from(auditLogsTable).orderBy(desc(auditLogsTable.id)).limit(6);
  const privacyLevel = await getPrivacyLevel(actor.id);
  const summary = {
    totalUsers: Number(userTotals.totalUsers),
    threatsDetected: Number(threatTotals.threatsDetected),
    highRiskEvents: Number(highTotals.highRiskEvents),
    averageTrustScore: Number(Number(userTotals.averageTrustScore ?? 0).toFixed(1)),
    privacyLevel,
    privacyStatus: `${privacyLevel} privacy protection active`,
    threatDistribution: Object.fromEntries(threatRows.map((row) => [row.classification, Number(row.total)])),
    riskDistribution: Object.fromEntries(riskRows.map((row) => [row.riskLevel, Number(row.total)])),
    recentActivity: activityRows.map((row) => ({
      id: row.id,
      eventType: row.eventType,
      description: row.description,
      riskLevel: riskFromTrust(Math.max(0, 100 - row.riskScore * 100)),
      timestamp: row.timestamp.toISOString(),
    })),
  };
  res.json(GetDashboardResponse.parse(summary));
});

export default router;